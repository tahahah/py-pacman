import json
import logging
import os
import pickle
import tracemalloc  # Add this import for memory profiling

import pika
import psutil  # Add this import to monitor memory usage
import redis
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from PIL import Image
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from redis.retry import Retry

# Load environment variables from .env file
load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(project="PacmanDataGen", job_type="worker", magic=True)

# Get HF_TOKEN from environment variables
HF_TOKEN = os.getenv('HF_TOKEN')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start memory profiling
tracemalloc.start()

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: RSS={mem_info.rss / (1024 * 1024)} MB, VMS={mem_info.vms / (1024 * 1024)} MB")
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    logger.info("Top 10 memory usage lines:")
    for stat in top_stats[:10]:
        logger.info(stat)

# Function to save buffered data to Hugging Face dataset in smaller batches
def save_to_hf_dataset(episodes_data, episode_keys):
    try:
        # Initialize lists to accumulate data
        all_episodes = []
        all_frames = []
        all_actions = []
        all_next_frames = []
        all_dones = []

        for data in episodes_data:
            episode = data['episode']
            frames_buffer = data['frames']
            actions_buffer = data['actions']
            next_frames_buffer = data['next_frames']
            dones_buffer = data['dones']

            logging.info(f"\tEpisode: {episode}")
            logging.info(f"\tNumber of frames: {len(frames_buffer)}, Number of actions: {len(actions_buffer)}")
            logging.info(f"\tNumber of next frames: {len(next_frames_buffer)}, Number of dones: {len(dones_buffer)}")
            logging.info("---------------------------------------------------------")

            # Convert frames to PIL images
            frames_buffer = [Image.fromarray(frame) for frame in frames_buffer]
            next_frames_buffer = [Image.fromarray(frame) for frame in next_frames_buffer]

            all_episodes.extend([episode] * len(frames_buffer))
            all_frames.extend(frames_buffer)
            all_actions.extend(actions_buffer)
            all_next_frames.extend(next_frames_buffer)
            all_dones.extend(dones_buffer)

        batch_dict = {
            'episode': all_episodes,
            'frame_image': all_frames,
            'action': all_actions,
            'next_frame_image': all_next_frames,
            'done': all_dones
        }

        # Create dataset with image column
        dataset = Dataset.from_dict(batch_dict)
        logging.info(f"Dataset size in MB: {dataset.data.nbytes / (1024 * 1024)}")
        dataset.push_to_hub('PacmanDataset_Redis_Try', split='train', token=HF_TOKEN)
        logger.info("\tSaved to Hugging Face dataset")
        logging.info("***********************************************************")

        
        # Delete keys from Redis after processing
        for key in episode_keys:
            redis_client.delete(key)

        # Free up memory
        del all_episodes
        del all_frames
        del all_actions
        del all_next_frames
        del all_dones
        del batch_dict
        del dataset

    except Exception as e:
        logger.error("Failed to save to Hugging Face dataset", exc_info=True)

# Dictionary to keep track of batches
batch_data = {}

# Redis client
redis_client = redis.StrictRedis(
            host='redis', 
            port=6379, 
            db=0, 
            decode_responses=False, 
            password="pacman", 
            health_check_interval=30, 
            socket_keepalive=True,
            retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
            retry_on_error=[ConnectionError, TimeoutError]
        )

def callback(ch, method, properties, body):
    print(f">>>>>>>>>>> Received message >>>>>>>>>>>")
    try:
        # Deserialize the message
        episode_keys = json.loads(body)
        logging.info(f"Received episode keys: {episode_keys}")

        episodes_data = []
        for key in episode_keys:
            # Deserialize the data using pickle
            data = pickle.loads(redis_client.get(key))
            episodes_data.append(data)

        # Save combined data to Hugging Face dataset
        save_to_hf_dataset(episodes_data, episode_keys)

        # Delete keys from Redis after processing
        for key in episode_keys:
            redis_client.delete(key)

        logging.info("<<<<<<<<<<< Processed message <<<<<<<<<<<")

    except Exception as e:
        logger.error("Failed to process message", exc_info=True)


def main():
    credentials = pika.PlainCredentials('worker', 'worker_pass')
    parameters = pika.ConnectionParameters(
        'rabbitmq-host',
        5672,
        '/',
        credentials
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='HF_upload_queue')
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(queue='HF_upload_queue', on_message_callback=callback, auto_ack=True)

    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    main()