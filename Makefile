# Laya container: build, run, test.
COMPOSE := docker compose
K ?= 4,8,16,32,77
N ?= 150

.PHONY: help volume build up down logs ready test bench smoke shell clean rebuild \
        eval eval-labels eval-probes eval-typed report introspect

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

eval: eval-probes eval-labels ## Run the full eval suite against the running checkpoint

eval-labels: up ## Accuracy + calibration vs number of choice labels (Banking77)
	$(COMPOSE) --profile eval run --rm eval labels --k $(K) --n $(N)

eval-probes: up ## Graded intensity ladders (churn, urgency)
	$(COMPOSE) --profile eval run --rm eval probes

eval-typed: ## Re-run the whole eval against the laya-typed-decisions checkpoint
	LAYA_SUBFOLDER=typed-decisions $(COMPOSE) up -d --force-recreate laya
	$(MAKE) ready
	LAYA_SUBFOLDER=typed-decisions $(COMPOSE) --profile eval run --rm eval all --k $(K) --n $(N)
	@echo "note: service is still on typed-decisions; 'make up' restores the base checkpoint"

report: ## Render a comparison table from everything in results/
	$(COMPOSE) --profile eval run --rm --entrypoint python eval -m eval.report

introspect: ## Print laya's real signatures (run after a version bump)
	docker run --rm -v laya-models:/models -v "$$PWD/scripts:/srv/scripts:ro" laya:cpu python scripts/introspect.py

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
