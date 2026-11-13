UV ?= uv
UV_RUN ?= $(UV) run --frozen
LLM_PROVIDER ?= litellm
BENCHMARK_DATASET ?= textoir_banking77
BENCHMARK_DIR ?= data/benchmarks/banking77-full
BENCHMARK_CONFIG ?= configs/pipeline_config.yaml
BENCHMARK_DOMAIN ?= generic
OOS_FILE ?=

.PHONY: ci-install ci-quality ci-docs ci-eval ci-benchmark lint format test

ci-install:
	$(UV) lock --check
	$(UV) sync --all-groups --frozen

ci-quality:
	$(UV) lock --check
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .
	$(UV_RUN) pytest tests -q --cov=src/ticket_classification --cov-fail-under=90 --cov-report=term-missing
	$(UV) run --frozen --extra discovery pytest tests/test_runtime_edges.py -q

ci-docs:
	$(UV_RUN) python scripts/check_docs.py

ci-eval: ci-install
	@if [ -n "$$LLM_GATEWAY_API_KEY" ] || [ -n "$$AI_EVAL_API_KEY" ] || [ -f .env ]; then \
		LLM_GATEWAY_API_KEY="$${LLM_GATEWAY_API_KEY:-$$AI_EVAL_API_KEY}" \
		$(UV_RUN) python -m ticket_classification.controllers.cli \
			--dataset "$(BENCHMARK_DATASET)" \
			--provider "$(LLM_PROVIDER)" \
			--max-samples 240 \
			--top-n-classes 8 \
			--k-min 6 \
			--k-max 10 \
			--report-path data/reports/benchmark.md; \
	else \
		$(UV_RUN) pytest tests/test_pipeline_and_mvc.py -k "test_end_to_end_service_and_fastapi_mvc_controllers" -q; \
	fi

ci-benchmark:
	@if [ -n "$$LLM_GATEWAY_API_KEY" ] || [ -n "$$AI_EVAL_API_KEY" ] || [ -f .env ]; then \
		if [ -n "$$AI_EVAL_API_KEY" ] && [ -z "$$LLM_GATEWAY_API_KEY" ]; then export LLM_GATEWAY_API_KEY="$$AI_EVAL_API_KEY"; fi; \
		if [ -z "$(OOS_FILE)" ]; then \
			if [ "$(BENCHMARK_DATASET)" != "textoir_banking77" ]; then echo "Set a domain-appropriate OOS_FILE for this dataset" >&2; exit 2; fi; \
			$(UV_RUN) python scripts/prepare_banking77_benchmark.py --output-dir "$(BENCHMARK_DIR)" --config "$(BENCHMARK_CONFIG)" || exit $$?; \
			set -- --config "$(BENCHMARK_DIR)/benchmark.yaml" --domain-id banking77 --oos-file "$(BENCHMARK_DIR)/oos-test.json"; \
		else \
			set -- --config "$(BENCHMARK_CONFIG)" --domain-id "$(BENCHMARK_DOMAIN)" --oos-file "$(OOS_FILE)"; \
		fi; \
		mkdir -p "$(BENCHMARK_DIR)"; \
		DATA_DIR="$(abspath $(BENCHMARK_DIR))/run-data" $(UV_RUN) python -m ticket_classification.controllers.cli \
			--dataset "$(BENCHMARK_DATASET)" --provider "$(LLM_PROVIDER)" --no-llm-naming "$$@" \
			--report-path "$(BENCHMARK_DIR)/full-benchmark.md" > "$(BENCHMARK_DIR)/full-benchmark.json"; \
	else \
		$(UV_RUN) pytest tests/test_pipeline_and_mvc.py -k "test_end_to_end_service_and_fastapi_mvc_controllers" -q; \
	fi

lint:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

test:
	$(UV_RUN) pytest -v
