# Cosmic-string-detection
Collection of codes to detect cosmic string the PTA datasets

In correlation_recovery one can generate the pulsar timing data with added signal from a cosmic string. There is 50 sample pulsars in sim_rand_gwb. The ouput files are saved in output_3. The pulsar pair angular distances and correlations are stored in out['xi'] and out['rho']/out["A2"]
Basic usage:
```
python inject_recovery_signal.py --datadir sim_rand_gwb/ --iter_num 1e5 --iter_real 1 --datadir_out output_3/ --sampler True --fgw 1e-8 --h 1e-14
```
In point_source_detection is the script which generates the signal with a point-source-like cosmic string. It also performs the recovery. Basic usage.
```
python string_inject_detect.py
```
The output files are in psrE1
