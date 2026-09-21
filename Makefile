# Laya container: build, run, test.
COMPOSE := docker compose
PLAYS ?= 120

.PHONY: help volume build up down logs ready test bench smoke shell clean rebuild \
        introspect agent-deps play play-mock play-watch solvable eval

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

introspect: ## Print laya's real signatures (run after a version bump)
	docker run --rm -v laya-models:/models -v "$$PWD/scripts:/srv/scripts:ro" laya:cpu python scripts/introspect.py

agent-deps: ## Link playwright for the game driver
	npm link playwright

play: up ## Autopilot plays Reindeer Jump against the live model (records video)
	node agent/play.mjs --decisions $(PLAYS) --out runs/laya

play-mock: ## Same harness, fake decision service -- proves the rig without the model
	node agent/play.mjs --mock --decisions $(PLAYS) --out runs/mock

solvable: ## Prove every wave has a passable, reachable lane (no model needed)
	node agent/check-solvable.mjs --waves 400

eval: up ## The measurements behind the prompt design
	$(COMPOSE) --profile eval run --rm --no-deps --entrypoint python eval -m eval.obstacle_class
	$(COMPOSE) --profile eval run --rm --no-deps --entrypoint python eval -m eval.lane_forced

play-watch: up ## Open a real browser and watch it play (needs a display)
	node agent/play.mjs --headed --decisions 0 --seconds 600 --video false

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
