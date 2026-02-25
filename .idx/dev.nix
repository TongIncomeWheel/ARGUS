# Google Project IDX Configuration for ARGUS
{ pkgs, ... }: {
  channel = "stable-23.11";
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
  ];
  idx = {
    extensions = [ "ms-python.python" ];
    workspace = {
      onCreate = {
        # Setup Virtual Env and Install Dependencies
        create-venv = ''
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
        '';
      };
      onStart = {
        activate-venv = "source .venv/bin/activate";
      };
    };
    previews = {
      enable = true;
      previews = {
        web = {
          command = ["streamlit" "run" "app.py" "--server.port" "$PORT" "--server.enableCORS" "false"];
          manager = "web";
        };
      };
    };
  };
}
