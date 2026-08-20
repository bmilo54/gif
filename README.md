**Windows**

```sql
psql -U postgres -h localhost
```

Then paste:

```sql
CREATE DATABASE gif_db;
CREATE USER gif WITH PASSWORD '2+PJh#&?';
ALTER ROLE gif SET client_encoding TO 'utf8';
ALTER ROLE gif SET default_transaction_isolation TO 'read committed';
ALTER ROLE gif SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE gif_db TO gif;
\c gif_db;
ALTER SCHEMA public OWNER TO gif;
GRANT ALL ON SCHEMA public TO gif;
\q