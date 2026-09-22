# Document classification suite. Self-contained: nothing here touches the game.
#
# The endpoint is LAYA_URL and defaults to a service on this machine. To use a
# faster one, pass it in -- no host is hard-coded in this repository:
#
#   make sweep LAYA_URL=http://<your-host>:8000
#
ROOT     := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))/..
IMAGE    ?= laya-tests:latest
LAYA_URL ?= http://127.0.0.1:8000
LAYA_READY_TIMEOUT ?= 900

# --network host so a LAYA_URL of 127.0.0.1 reaches a service published on this
# machine, while an external URL keeps working unchanged.
RUN = docker run --rm --network host \
        -e LAYA_URL=$(LAYA_URL) -e LAYA_READY_TIMEOUT=$(LAYA_READY_TIMEOUT) \
        -e PYTHONPATH=/work -w /work \
        -v "$(ROOT)/classify:/work/classify" $(IMAGE)

.PHONY: help image test smoke sweep report all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n",$$1,$$2}'
	@echo
	@echo "  endpoint: $(LAYA_URL)   (override with LAYA_URL=...)"

image: ## Build the runner image (pytest + httpx, no model)
	@docker image inspect $(IMAGE) >/dev/null 2>&1 || \
	  docker build -f "$(ROOT)/Dockerfile.tests" -t $(IMAGE) "$(ROOT)"

test: image ## Offline tests only: corpus integrity and metrics. No service needed.
	$(RUN) pytest classify/tests/test_corpus.py classify/tests/test_metrics.py

smoke: test ## Offline tests, then a small live slice and a few dozen sweep calls
	$(RUN) pytest classify/tests/test_smoke.py
	$(RUN) python -m classify.sweep --smoke

sweep: ## The full sweep. Slow on CPU -- point LAYA_URL at faster hardware.
	$(RUN) python -m classify.sweep

report: ## Render results/*.json into report.md
	$(RUN) python -m classify.report

# Tests first, and the report only once they pass: `make` stops at the first
# failing recipe, so a red suite never produces a report to be quoted from.
all: test smoke sweep report ## Everything, in order, stopping on the first failure

clean: ## Remove generated results and the rendered report
	rm -f "$(ROOT)/classify/results"/*.json "$(ROOT)/classify/results/report.md"
