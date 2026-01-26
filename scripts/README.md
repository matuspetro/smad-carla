# CARLA Simulator Startup Script

This PowerShell script, `start_carla.ps1`, is designed to simplify the process of starting the CARLA simulator with customizable options.

### Prerequisites
- Ensure PowerShell is installed on your system.
- Verify that the CARLA simulator is downloaded and extracted to a directory on your machine.

#### Start CARLA with Default Settings
```powershell
.\start_carla.ps1
```

#### Start CARLA on a Custom Port
```powershell
.\start_carla.ps1 -port 3000
```

#### Start CARLA in Low-Quality Mode
```powershell
.\start_carla.ps1 -lowQuality
```

#### Start CARLA from a Custom Path
```powershell
.\start_carla.ps1 -carlaPath "D:\Simulators\CARLA" -port 3000 -lowQuality
```

## Notes
- Ensure the `CarlaUE4.exe` file exists in the specified `carlaPath`.
- The script uses `Start-Process` to launch the simulator in a new process.