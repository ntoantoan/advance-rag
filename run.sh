docker compose down -v
docker compose up

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/app.py