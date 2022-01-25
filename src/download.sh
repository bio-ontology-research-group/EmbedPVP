perl annotate_variation.pl -downdb 1000g2015aug humandb -buildver hg38
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 gnomad  humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 exac03 humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 refGene  humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 dbnsfp30a humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 mcap  humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 dbscsnv11 humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 clinvar_20190305  humandb/
perl annotate_variation.pl -downdb -webfrom annovar -build hg38 revel humandb/
