The `gmt_enc_v1.0.0_20261020` policy requires gravity compensation for the arm joints. Before running this policy, update the following configuration files on the robot.

1. Enable the existing `GravityCompensationModule` entry in `/robros/configs/robot.yaml`:

   ```yaml
   GravityCompensationModule:
     enabled: true
   ```

2. Set `gravity_comp_gain` in `/robros/configs/controller/params.yaml` to the following values:

   ```yaml
   gravity_comp_gain: [
     0.0,
     0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0
   ]
   ```

Preserve the existing YAML nesting when updating these entries. The gain array is ordered by joint index; keep all 31 values in the order shown.
