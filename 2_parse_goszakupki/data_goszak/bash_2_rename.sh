input="/media/xenakas/ext4_drive/data_goszak/list_f.csv"
while IFS=, read -r line1 line2
do
#	printf "s3_auto/$line1 s4_auto/$line2" 
	cp /media/xenakas/ext4_drive/data_goszak/s3_auto/$line1 /media/xenakas/ext4_drive/data_goszak/s4_auto/$line2 
done < "$input"
