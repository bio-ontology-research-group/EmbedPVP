# To run: perl MAF_filter.pl annotated_vcf.vcf 0.01

use strict;

my ($test, $line, @arr, @result, $maf, $value);

if ($ARGV[0] eq "") {print "provide path to the input file - vcf format"; exit;}
if ($ARGV[1] eq "") {print "enter MAF value to filter"; exit;} #default=0.01
else {$maf=$ARGV[1];}
open IN, $ARGV[0] or die "cannot open the input file";

while ($line=<IN>)
{ chop $line; @result=();
@arr=split('\t', $line);

while ($arr[7] =~/1000g2015aug_afr=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/1000g2015aug_amr=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/1000g2015aug_eas=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/1000g2015aug_eur=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/1000g2015aug_sas=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/1000g2015aug_all=(.*?);/g) {push (@result,$1);}


while ($arr[7] =~/ExAC_AFR=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_AMR=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_EAS=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_FIN=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_NFE=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_OTH=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_SAS=(.*?);/g) {push (@result,$1);}
while ($arr[7] =~/ExAC_ALL=(.*?);/g) {push (@result,$1);}

while ($arr[7] =~/AF_raw=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_male=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_female=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_afr=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_ami=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_amr=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_asj=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_eas=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_fin=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_nfe=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_oth=(.*?);/g) {push(@result,$1);}
while ($arr[7] =~/AF_sas=(.*?);/g) {push(@result,$1);}


$test=0;
foreach $value (@result)
{
if (($value<$maf) or ($value eq  "."))
   {$test=1;}
else {$test=0; goto nextt;}      
}

if ( ($test eq 1) and ($line!~/0\/0/) ) {print $line."\n";}

nextt:

@result=();
}
close IN;
