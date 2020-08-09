#!/bin/bash

export PGPASSWORD=test-task

docker run \
  --name test-task-postgres \
  -e POSTGRES_PASSWORD=${PGPASSWORD} \
  -p 8213:5432 \
  -d postgres:9.4

psql \
  -d postgres \
  --host=localhost \
  --port 8213 \
  -U postgres \
  -f db.sql

# sanituy check
container_name=test-task-postgres
if [ "$( docker container inspect -f '{{.State.Running}}' $container_name )" == "true" ]; then
    echo "OK: Container is running"
else 
    echo "ERROR: Please, check the container"
fi
