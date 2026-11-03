setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .
validate:
	python scripts/validate_data.py
train:
	python -m incident_risk.train
test:
	pytest -q
lint:
	ruff check . && ruff format --check .
api:
	uvicorn api.main:app --reload --port 8000
dash:
	streamlit run dashboard/app.py
docker:
	docker compose up --build
tune:
	python -m incident_risk.tune
calib:
	python -m incident_risk.calibration_compare
champion:
	python -m incident_risk.champion_report
ablate:
	python -m incident_risk.experiments --model lightgbm
register:
	python -m incident_risk.register
reproduce:
	make validate && make tune && make train && make champion && make register
explain:
	python -m incident_risk.explain
robust:
	python -m incident_risk.robustness
subgroup:
	python -m incident_risk.subgroup
uncertainty:
	python -m incident_risk.uncertainty
governance:
	make explain robust subgroup uncertainty