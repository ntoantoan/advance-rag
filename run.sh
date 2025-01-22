docker compose down -v && docker compose up -d && export PYTHONPATH=$PYTHONPATH:$(pwd)/src && python src/app.py
