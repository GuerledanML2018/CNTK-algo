nb_image=`ls -l *.jp*g | wc -l`
i=0
for f in *.jp*g;
do convert $f -resize !128x72 $f && i=`python -c "print $i+1"` && pourcentage=`python -c "print float($i)/$nb_image"` && echo "image $i/$nb_image -> $pourcentage %";
done;
echo "Toutes les images ont été traitées ($nb_image)"

