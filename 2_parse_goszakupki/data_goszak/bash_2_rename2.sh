input="/media/xenakas/ext4_drive/data_goszak/list_f2.csv"
while IFS=, read -r line1 line2
do
#	printf "s3_auto/$line1 s4_auto/$line2" 
	cp /media/xenakas/ext4_drive/data_goszak/1_data_goszak_original/$line1 /media/xenakas/ext4_drive/data_goszak/2_data_goszak_renamed/$line2 
done < "$input"
