#!/bin/sh



# Define RabbitMQ user and password for pacman
RABBITMQ_USER_PACMAN=pacman
RABBITMQ_PASSWORD_PACMAN=pacman_pass

# Define RabbitMQ user and password for worker
RABBITMQ_USER_WORKER=worker
RABBITMQ_PASSWORD_WORKER=worker_pass

# Ensure the nodename doesn't change, e.g. if docker restarts.
# Important because rabbitmq stores data per node name (or 'IP')
echo 'NODENAME=rabbit@localhost' > /etc/rabbitmq/rabbitmq-env.conf

# Set the maximum message size
echo 'max_message_size = 104857600' >> /etc/rabbitmq/rabbitmq.conf

# Create Rabbitmq users
(rabbitmqctl wait --timeout 60 $RABBITMQ_PID_FILE ; \
rabbitmqctl add_user $RABBITMQ_USER_PACMAN $RABBITMQ_PASSWORD_PACMAN 2>/dev/null ; \
rabbitmqctl set_user_tags $RABBITMQ_USER_PACMAN administrator ; \
rabbitmqctl set_permissions -p / $RABBITMQ_USER_PACMAN  ".*" ".*" ".*" ; \
echo "*** User '$RABBITMQ_USER_PACMAN' with password '$RABBITMQ_PASSWORD_PACMAN' completed. ***" ; \
rabbitmqctl add_user $RABBITMQ_USER_WORKER $RABBITMQ_PASSWORD_WORKER 2>/dev/null ; \
rabbitmqctl set_user_tags $RABBITMQ_USER_WORKER administrator ; \
rabbitmqctl set_permissions -p / $RABBITMQ_USER_WORKER  ".*" ".*" ".*" ; \
echo "*** User '$RABBITMQ_USER_WORKER' with password '$RABBITMQ_PASSWORD_WORKER' completed. ***" ; \
echo "*** Log in the WebUI at port 15672 (example: http:/localhost:15672) ***") &

# $@ is used to pass arguments to the rabbitmq-server command.
# For example if you use it like this: docker run -d rabbitmq arg1 arg2,
# it will be as you run in the container rabbitmq-server arg1 arg2
rabbitmq-server $@