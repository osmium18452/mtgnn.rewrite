output_list="720 480 360 320 192 160 96 80 48 40 24 20"
dataset_list="gweather etth1 etth2 ettm1 ettm2 exchange wht finance"
batch_size=256
stride=1
epochs=20
exec_date=23.7.27.formal

dataset_count=1
output_count=1
cuda="3"

for dataset in $dataset_list; do
    for output_len in $output_list; do
        echo "\033[35mmodel: dataset: $dataset_count/8, output len: $output_count/12\033[0m"
        save_file="save/$exec_date/best/$dataset/$output_len"
        exec="python main.py --fudan -GBDC $cuda -e $epochs -o $output_len -b $batch_size --fixed_seed 3407 -d $dataset -S $save_file --draw --stride $stride"
        echo $exec
        if [ ! -e $save_file/result.json ]; then
            $exec
        else
            echo 'file exists'
        fi
        save_file="save/$exec_date/worst/$dataset/$output_len"
        exec="python main.py --fudan -GDC $cuda -e $epochs -o $output_len -b $batch_size --fixed_seed 3407 -d $dataset -S $save_file --stride $stride"
        echo $exec
        if [ ! -e $save_file/result.json ]; then
            $exec
        else
            echo 'file exists'
        fi
        output_count=$(expr $output_count + 1)
    done
    output_count=1
    dataset_count=$(expr $dataset_count + 1)
done

output_list="24 36 48 60"
dataset="ill"
for output_len in $output_list; do
    save_file="save/$exec_date/best/$dataset/$output_len"
    exec="python main.py --fudan -GBDC $cuda -e $epochs -o $output_len -b $batch_size --fixed_seed 3407 -d $dataset -S $save_file --draw --stride $stride"
    echo $exec
    if [ ! -e $save_file/result.json ]; then
        $exec
    else
        echo 'file exists'
    fi
    save_file="save/$exec_date/worst/$dataset/$output_len"
    exec="python main.py --fudan -GDC $cuda -e $epochs -o $output_len -b $batch_size --fixed_seed 3407 -d $dataset -S $save_file --stride $stride"
    echo $exec
    if [ ! -e $save_file/result.json ]; then
        $exec
    else
        echo 'file exists'
    fi
    output_count=$(expr $output_count + 1)
done
