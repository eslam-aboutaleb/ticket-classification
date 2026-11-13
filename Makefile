UV ?= uv
UV_RUN ?= $(UV) run --frozen
LLM_PROVIDER ?= openai
BENCHMARK_DATASET ?= textoir_banking77

.PHONY: ci-install ci-quality ci-docs ci-eval ci-benchmark lint format test

ci-install:
	$(UV) sync --all-groups --frozen

ci-quality:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .
	$(UV_RUN) pytest tests -q --cov=src/ticket_classification --cov-fail-under=85 --cov-report=term-missing

ci-docs:
	$(UV_RUN) python -c "from pathlib import Path; import yaml; from ticket_classification.config.settings import LLMProvider; from ticket_classification.config.pipeline import PipelineConfig; \
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
		$(UV_RUN) python -m ticket_classification.controllers.cli \
			--dataset "$(BENCHMARK_DATASET)" \
			--provider "$(LLM_PROVIDER)" \
			--max-samples 240 \
			--top-n-classes 8 \
			--k-min 6 \
			--k-max 10; \
	else \
		$(UV_RUN) pytest tests/test_pipeline_and_mvc.py -k "test_end_to_end_service_and_fastapi_mvc_controllers" -q; \
	fi

ci-benchmark:
	@if [ -n "$$LLM_GATEWAY_API_KEY" ] || [ -n "$$AI_EVAL_API_KEY" ] || [ -f .env ]; then \
		LLM_GATEWAY_API_KEY="$${LLM_GATEWAY_API_KEY:-$$AI_EVAL_API_KEY}" \
		$(UV_RUN) python -m ticket_classification.controllers.cli \
			--dataset "$(BENCHMARK_DATASET)" \
			--provider "$(LLM_PROVIDER)" \
			--max-samples 240 \
			--top-n-classes 8 \
			--k-min 6 \
			--k-max 10; \
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
