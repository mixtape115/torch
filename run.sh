xhost +

docker run -it \
    --env="DISPLAY=${DISPLAY}" \
    -p 8889:8889 \
    --gpus all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${PWD}/src/:/home/user/src/" \
    --name="torch" \
    torch \
    bash
echo "done"
