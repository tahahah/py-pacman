import logging
import os
import pickle

import pika
from datasets import Dataset
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Get HF_TOKEN from environment variables
HF_TOKEN = os.getenv('HF_TOKEN')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to save buffered data to Hugging Face dataset in smaller batches
def save_to_hf_dataset(data):
    try:
        # Extract data from the dictionary
        episode = data['episode']
        frame = data['frame']
        frames_buffer = data['frames']
        actions_buffer = data['actions']
        next_frames_buffer = data['next_frames']
        dones_buffer = data['dones']
        
        # Convert frames to PIL images
        frames_buffer = [Image.fromarray(frame) for frame in frames_buffer]
        next_frames_buffer = [Image.fromarray(frame) for frame in next_frames_buffer]
        
        batch_dict = {
            'episode': [episode] * len(frames_buffer),
            'frame': [frame] * len(frames_buffer),
            'frame_image': frames_buffer,
            'action': actions_buffer,
            'next_frame_image': next_frames_buffer,
            'done': dones_buffer
        }
        
        # Create dataset with image column
        dataset = Dataset.from_dict(batch_dict)
        dataset = dataset.cast_column('frame_image', Image())
        dataset = dataset.cast_column('next_frame_image', Image())
        
        dataset.push_to_hub('pacman_dataset_gamengen_1', split='train', token=HF_TOKEN)
        logger.info("Saved to Hugging Face dataset")
    except Exception as e:
        logger.error("Failed to save to Hugging Face dataset", exc_info=True)

# Dictionary to keep track of batches
batch_data = {}

def callback(ch, method, properties, body):
    print(f"Received {body}")
    try:
        # Deserialize the message
        data = pickle.loads(body)
        episode = data['episode']
        batch_id = data['batch_id']
        is_last_batch = data['is_last_batch']

        # Initialize episode entry if not exists
        if episode not in batch_data:
            batch_data[episode] = {}

        # Store batch data
        batch_data[episode][batch_id] = data

        # Check if all batches are received
        if is_last_batch:
            combined_data = {
                'episode': episode,
                'frame': 0,
                'frames': [],
                'actions': [],
                'next_frames': [],
                'dones': []
            }
            for batch_id in sorted(batch_data[episode].keys()):
                batch = batch_data[episode][batch_id]
                combined_data['frame'] += batch['frame']
                combined_data['frames'].extend(batch['frames'])
                combined_data['actions'].extend(batch['actions'])
                combined_data['next_frames'].extend(batch['next_frames'])
                combined_data['dones'].extend(batch['dones'])

            # Save combined data to Hugging Face dataset
            save_to_hf_dataset(combined_data)
            ch.basic_ack(delivery_tag=method.delivery_tag)


            # Clear stored data for the episode
            del batch_data[episode]

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