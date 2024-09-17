#!/bin/bash

set -e
set -u

function create_user_and_database_with_extension() {
	local database=$1
	echo "  Creating user and database '$database'"
	psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
	    CREATE USER $database;
	    CREATE DATABASE $database;
	    GRANT ALL PRIVILEGES ON DATABASE $database TO $database;
EOSQL
  echo "  Creating vector extension for '$database'"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" $database <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL
}

if [ -n "$DATABASES" ]; then
	echo "Multiple database creation requested: $DATABASES"
	for db in $(echo $DATABASES | tr ',' ' '); do
		create_user_and_database_with_extension $db
	done
	echo "Multiple databases created"
fi
