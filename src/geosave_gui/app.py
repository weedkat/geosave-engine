from pathlib import Path

import streamlit as st


# ---------- Data helpers ----------

def get_project_root() -> Path:
	return Path(__file__).resolve().parent.parent / "geosave_engine" / "_project"


def list_projects() -> list[str]:
	root = get_project_root()
	if not root.exists():
		return []

	projects: list[str] = []
	for domain in sorted(p for p in root.iterdir() if p.is_dir()):
		for project in sorted(p for p in domain.iterdir() if p.is_dir()):
			projects.append(f"{domain.name}/{project.name}")
	return projects


def list_configs(project: str) -> list[str]:
	cfg_dir = get_project_root() / project / "configs"
	if not cfg_dir.exists():
		return []
	return [p.name for p in sorted(cfg_dir.glob("*.yaml"))]


# ---------- UI pages ----------

def page_home() -> None:
	st.title("GeoSave")
	st.write("Simple Streamlit starter for your workflow.")
	st.write("Use the sidebar to open Projects, Train, Predict, or Runs.")


def page_projects(projects: list[str]) -> None:
	st.header("Projects")
	if not projects:
		st.warning("No projects found yet.")
		return

	st.write("Available projects:")
	st.dataframe({"project": projects}, hide_index=True, use_container_width=True)


def page_train(selected_project: str | None) -> None:
	st.header("Train")
	if not selected_project:
		st.info("Pick a project in the sidebar first.")
		return

	configs = list_configs(selected_project)
	if not configs:
		st.warning("No YAML configs found for this project.")
		return

	cfg = st.selectbox("Training config", configs)
	if st.button("Start training", type="primary"):
		st.success(f"TODO: start training with {selected_project} / {cfg}")


def page_predict(selected_project: str | None) -> None:
	st.header("Predict")
	if not selected_project:
		st.info("Pick a project in the sidebar first.")
		return

	configs = list_configs(selected_project)
	if not configs:
		st.warning("No YAML configs found for this project.")
		return

	cfg = st.selectbox("Prediction config", configs)
	if st.button("Run prediction", type="primary"):
		st.success(f"TODO: run prediction with {selected_project} / {cfg}")


def page_runs() -> None:
	st.header("Runs")
	st.info("TODO: show training/prediction history here.")


# ---------- App entry ----------

def main() -> None:
	st.set_page_config(page_title="GeoSave", layout="wide")

	projects = list_projects()

	st.sidebar.title("Menu")
	page = st.sidebar.radio("Go to", ["Home", "Projects", "Train", "Predict", "Runs"])

	selected_project = None
	if projects:
		selected_project = st.sidebar.selectbox("Project", ["-"] + projects)
		if selected_project == "-":
			selected_project = None

	if page == "Home":
		page_home()
	elif page == "Projects":
		page_projects(projects)
	elif page == "Train":
		page_train(selected_project)
	elif page == "Predict":
		page_predict(selected_project)
	else:
		page_runs()


if __name__ == "__main__":
	main()
