.PHONY: up dev ingest logs down

up:
	docker compose up --build

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

ingest:
	docker compose run --rm backend python -m app.scripts.ingest

logs:
	docker compose logs -f

down:
	docker compose down
