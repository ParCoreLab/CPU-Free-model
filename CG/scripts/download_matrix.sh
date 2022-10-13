MATRIX_LINK=${MATRIX_LINK:-NO_LINK_PROVIDED}
DOWNLOAD_PATH=${DOWNLOAD_PATH:-/global/D1/homes/iismayilov/matrices/}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

if [[ $MATRIX_LINK == NO_LINK_PROVIDED ]] ; then
  echo "Please provide a link to a SuiteSparse matrix"
  exit
fi


wget $MATRIX_LINK -O - | tar -xvz -C $DOWNLOAD_PATH --strip-components=1
