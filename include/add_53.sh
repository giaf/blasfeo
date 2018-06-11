find ./ -type f -not -path "*git*" -not -path "*add_53*" -exec sed -i 's/defined(TARGET_ARMV8A_ARM_CORTEX_A57)/defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)/g' {} \;
find ./ -type f -not -path "*git*" -not -path "*add_53*" -exec sed -i 's/defined( TARGET_ARMV8A_ARM_CORTEX_A57 )/defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)/g' {} \;
