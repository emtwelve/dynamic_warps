cd tests/$1
rm -rf objs/

#Remove the generated files
rm opt_$1.cu

#Remove the log files
rm log.csv
rm warp.log

# Remove the executables
rm remap_warp
rm $1
rm opt_$1
rm anno_$1
