#AF annotation
input=$1
output=$2
path_to_annovar=$3
perl $path_to_annovar/table_annovar.pl \
   $input  $path_to_annovar/humandb/ \
    -buildver hg38 -out output -remove -protocol knownGene,1000g2015aug_all,1000g2015aug_amr,1000g2015aug_afr,1000g2015aug_eas,1000g2015aug_eur,1000g2015aug_sas,exac03,gnomad30_genome -operation g,f,f,f,f,f,f,f,f -nastring . -vcfinput
