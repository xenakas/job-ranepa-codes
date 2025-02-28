input="/media/xenakas/ext4_drive/data_goszak/list_of_goszak"
while IFS= read -r line
do
	cp s3_alexander/$line s3_auto/$line 
done < "$input"
echo $newname
