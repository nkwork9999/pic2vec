-- pic2vec + DuckDB vss image search template.
--
-- Usage:
--   INSTALL vss;
--   LOAD vss;
--   LOAD pic2vec;
--   .read examples/vss_image_search.sql
--
-- Then replace the glob path and query image path below.

CREATE OR REPLACE MACRO pic_embed_192(path) AS (
    pic_embed(path)::FLOAT[192]
);

CREATE OR REPLACE TABLE pic2vec_images (
    filename VARCHAR,
    vec FLOAT[192]
);

INSERT INTO pic2vec_images
SELECT filename, pic_embed_192(filename) AS vec
FROM glob('/path/to/images/*.{jpg,jpeg,png}');

CREATE INDEX pic2vec_images_hnsw
ON pic2vec_images
USING HNSW (vec)
WITH (metric = 'cosine');

SELECT filename,
       array_cosine_distance(vec, pic_embed_192('/path/to/query.jpg')) AS distance
FROM pic2vec_images
ORDER BY distance
LIMIT 10;
