UV ?= uv
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || command -v python; fi)
LLM_PROVIDER ?= openai
BENCHMARK_DATASET ?= textoir_banking77

.PHONY: ci-install ci-quality ci-docs ci-eval ci-benchmark lint format test

ci-install:
	@if [ -d .venv ]; then \
		$(UV) pip install --python .venv/bin/python -e ".[dev]"; \
	else \
		$(UV) pip install --system --python "$$(python3 -c 'import sys; print(sys.executable)')" -e ".[dev]"; \
	fi

ci-quality:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m pytest tests -q --cov=src/ticket_classification --cov-fail-under=85 --cov-report=term-missing

ci-docs:
	$(PYTHON) -c "from pathlib import Path; import yaml; from ticket_classification.config.settings import LLMProvider; from ticket_classification.config.pipeline import PipelineConfig; \
	plan = Path('PLAN.md').read_text(encoding='utf-8'); \
	assert 'llm-gateway-sdk' in plan and 'DeepAligned' in plan and 'BERTopic' in plan, 'PLAN.md missing core sections'; \
	cfg = yaml.safe_load(Path('configs/pipeline_config.yaml').read_text(encoding='utf-8')); \
	PipelineConfig.model_validate(cfg); PipelineConfig.model_validate(yaml.safe_load(Path('configs/profiles.example.yaml').read_text(encoding='utf-8'))); \
	assert {'textoir_banking77', 'textoir_clinc_oos', 'cfpb_claims'} <= set(cfg['datasets']), 'Missing datasets in pipeline_config.yaml'; \
	assert {p.value for p in LLMProvider} == {'litellm', 'openai', 'gemini'}, 'Missing provider in LLMProvider'; \
	print('Documentation and configuration contracts verified.')"

# ai-eval.yml runs this target WITHOUT a prior ci-install, so install first.
ci-eval: ci-install
	@if [ -n "$$LLM_GATEWAY_API_KEY" ] || [ -n "$$AI_EVAL_API_KEY" ] || [ -f .env ]; then \
		LLM_GATEWAY_API_KEY="$${LLM_GATEWAY_API_KEY:-$$AI_EVAL_API_KEY}" \
		$(PYTHON) -m ticket_classification.controllers.cli \
			--dataset "$(BENCHMARK_DATASET)" \
			--provider "$(LLM_PROVIDER)" \
			--max-samples 240 \
			--top-n-classes 8 \
			--k-min 6 \
			--k-max 10; \
	else \
		$(PYTHON) -m pytest tests/test_pipeline_and_mvc.py -k "test_end_to_end_service_and_fastapi_mvc_controllers" -q; \
	fi

ci-benchmark:
	@if [ -n "$$LLM_GATEWAY_API_KEY" ] || [ -n "$$AI_EVAL_API_KEY" ] || [ -f .env ]; then \
		LLM_GATEWAY_API_KEY="$${LLM_GATEWAY_API_KEY:-$$AI_EVAL_API_KEY}" \
		$(PYTHON) -m ticket_classification.controllers.cli \
			--dataset "$(BENCHMARK_DATASET)" \
			--provider "$(LLM_PROVIDER)" \
			--max-samples 240 \
			--top-n-classes 8 \
			--k-min 6 \
			--k-max 10; \
	else \
		$(PYTHON) -m pytest tests/test_pipeline_and_mvc.py -k "test_end_to_end_service_and_fastapi_mvc_controllers" -q; \
	fi

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest -v
