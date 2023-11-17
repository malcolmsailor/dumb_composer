<!-- TODO 2023-11-16 update run commands to include output folder which is now required -->

To build ornamented chorales:

`python scripts/run_incremental_contrapuntist_with_prefabs.py /Users/malcolm/output/rncollage/chorales_output/*.txt --contrapuntist-config settings/chorale_interval_weights.yaml --prefab-config settings/chorales/prefab_applier_settings.yaml --seed 123`

To build ornamented chorales from Mozart:

`python scripts/run_incremental_contrapuntist_with_prefabs.py /Users/malcolm/output/rncollage/mozart_ps_output/*.txt --contrapuntist-config settings/chorale_interval_weights.yaml --prefab-config settings/chorales/prefab_applier_settings.yaml --seed 123 --num-workers 0`

To build un-ornamented:

`python scripts/run_incremental_contrapuntist.py /Users/malcolm/output/rncollage/mozart_ps_output/*.txt --contrapuntist-config settings/chorale_interval_weights.yaml --seed 123`

Build ornamented w/ accompaniment:

`python scripts/run_incremental_contrapuntist_with_prefabs_and_accomps.py /Users/malcolm/output/rncollage/mozart_ps_output/*.txt --contrapuntist-config settings/chorale_interval_weights.yaml --seed 123`

Build accompaniments only:

`python scripts/run_incremental_contrapuntist_with_accomps.py /Users/malcolm/output/rncollage/mozart_ps_output/*.txt --contrapuntist-config settings/chorale_interval_weights.yaml --seed 123 --num-workers 0 --debug`
