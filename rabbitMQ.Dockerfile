FROM rabbitmq

# Define environment variables.
ENV RABBITMQ_USER user
ENV RABBITMQ_PASSWORD user
ENV RABBITMQ_PID_FILE /var/lib/rabbitmq/mnesia/rabbitmq

ADD rabbit_init.sh /init.sh
RUN chmod +x /init.sh


# Expose RabbitMQ ports
EXPOSE 5672 15672

# Define default command
CMD ["/init.sh"]
