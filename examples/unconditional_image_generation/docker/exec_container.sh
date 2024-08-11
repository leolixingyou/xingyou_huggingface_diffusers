docker_run_command=$(cat run_container.sh)

docker_run_suffix=$(echo "$docker_run_command" | grep -oP "(?<=--name ).*"| awk '{print $1}')

xhost +
docker exec -it "$docker_run_suffix" bash

