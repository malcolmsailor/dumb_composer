#!/bin/env bash

set -e
SEED=123

# Counts

output_folder=${OUTPUT_DIR}/dumb_composer_synthetic_data

if [[ -d "$output_folder" ]]; then
    backup_path="${output_folder}_$(date -u +%Y-%m-%dT%H_%M_%SZ).zip"
    zip -r "${backup_path}" "${output_folder}"
    trash "${output_folder}"
fi

declare -A chorales
chorales[contrapuntist_n]=100
chorales[prefab_n]=200
chorales[accomp_n]=200
chorales[prefab_accomp_n]=400
chorales[rntxt_folder]=${OUTPUT_DIR}/rncollage/chorales_output

declare -A mozart
mozart[contrapuntist_n]=400
mozart[prefab_n]=800
mozart[accomp_n]=800
mozart[prefab_accomp_n]=1600
mozart[rntxt_folder]=${OUTPUT_DIR}/rncollage/mozart_ps_output

declare -A beethoven
beethoven[contrapuntist_n]=200
beethoven[prefab_n]=400
beethoven[accomp_n]=400
beethoven[prefab_accomp_n]=800
beethoven[rntxt_folder]=${OUTPUT_DIR}/rncollage/beethoven_output

names=("chorales" "mozart" "beethoven")

common_args=("--seed" $SEED "--shuffle-input-paths" "--output-folder" "${output_folder}" "--num-workers" 16)

for name in "${names[@]}"; do
    declare -n current_array="$name"
    python scripts/run_incremental_contrapuntist.py \
        "${current_array[rntxt_folder]}"/*.txt \
        --contrapuntist-config settings/chorale_interval_weights.yaml \
        --max-files ${current_array[contrapuntist_n]} \
        --basename-prefix "${name}" \
        "${common_args[@]}"

    python scripts/run_incremental_contrapuntist_with_prefabs.py \
        "${current_array[rntxt_folder]}"/*.txt \
        --contrapuntist-config settings/chorale_interval_weights.yaml \
        --prefab-config settings/chorales/prefab_applier_settings.yaml \
        --max-files ${current_array[prefab_n]} \
        --basename-prefix "${name}" \
        "${common_args[@]}"

    python scripts/run_incremental_contrapuntist_with_accomps.py \
        "${current_array[rntxt_folder]}"/*.txt \
        --contrapuntist-config settings/chorale_interval_weights.yaml \
        --max-files ${current_array[accomp_n]} \
        --basename-prefix "${name}" \
        "${common_args[@]}"

    python scripts/run_incremental_contrapuntist_with_prefabs_and_accomps.py \
        "${current_array[rntxt_folder]}"/*.txt \
        --contrapuntist-config settings/chorale_interval_weights.yaml \
        --max-files ${current_array[prefab_accomp_n]} \
        --basename-prefix "${name}" \
        "${common_args[@]}"

done
