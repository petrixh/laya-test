# Laya container: build, run, test.
COMPOSE := docker compose

.PHONY: help volume build up down logs ready test bench smoke shell clean rebuild

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

volume: ## Create the shared weights cache volume
	@docker volume inspect laya-models >/dev/null 2>&1 || docker volume create laya-models

build: volume ## Build the service image
	$(COMPOSE) build

up: volume ## Start the service (first run downloads ~1.7GB of weights)
	$(COMPOSE) up -d

ready: ## Block until the checkpoint is loaded
	@echo "waiting for /readyz ..."
	@for i in $$(seq 1 90); do \
	  if [ "$$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/readyz)" = "200" ]; then \
	    curl -s http://127.0.0.1:8000/info; echo; exit 0; fi; \
	  sleep 10; done; \
	echo "timed out"; $(COMPOSE) logs --tail=40 laya; exit 1

test: up ## Run the HTTP test suite in a container
	$(COMPOSE) --profile test run --rm --build tests

bench: up ## Report latency at 1 / 3 / 10 questions per pass
	$(COMPOSE) --profile test run --rm --entrypoint python tests bench.py

smoke: ## Load + predict directly in-process, bypassing HTTP
	docker run --rm -v laya-models:/models -v "$$PWD/scripts:/srv/scripts:ro" laya:cpu python scripts/smoke.py

logs: ## Tail service logs
	$(COMPOSE) logs -f laya

shell: ## Shell into the running service
	$(COMPOSE) exec laya bash

down: ## Stop the service (weights volume survives)
	$(COMPOSE) down

clean: down ## Stop and delete the weights volume too
	-docker volume rm laya-models

rebuild: ## Rebuild from scratch
	$(COMPOSE) build --no-cache
