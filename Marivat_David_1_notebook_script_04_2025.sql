-- requête 1
SELECT 
    o.order_id, 
    o.order_status, 
    o.order_purchase_timestamp, 
    o.order_delivered_customer_date, 
    o.order_estimated_delivery_date
FROM 
    orders o
WHERE 
    o.order_status != 'canceled'
    AND o.order_purchase_timestamp >= (
        SELECT DATE(MAX(order_purchase_timestamp), '-3 months')
        FROM orders
    )
    AND JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_estimated_delivery_date) >= 3;

-- requête 2
SELECT oi.seller_id, SUM(oi.price) AS total_revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY oi.seller_id
HAVING total_revenue > 100000;

-- requête 3
WITH max_order_date AS (
    SELECT MAX(order_purchase_timestamp) AS max_date
    FROM orders
),
first_sale_per_seller AS (
    SELECT 
        oi.seller_id,
        MIN(o.order_purchase_timestamp) AS first_sale_date
    FROM 
        order_items oi
        JOIN orders o ON oi.order_id = o.order_id
    GROUP BY 
        oi.seller_id
),
sales_last_3_months AS (
    SELECT 
        oi.seller_id,
        COUNT(*) AS products_sold
    FROM 
        order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN max_order_date m ON 1=1
    WHERE 
        o.order_purchase_timestamp >= DATE(m.max_date, '-3 months')
    GROUP BY 
        oi.seller_id
)
SELECT 
    fs.seller_id,
    fs.first_sale_date,
    sl.products_sold
FROM 
    first_sale_per_seller fs
    JOIN sales_last_3_months sl ON fs.seller_id = sl.seller_id
    JOIN max_order_date m ON 1=1
WHERE 
    fs.first_sale_date >= DATE(m.max_date, '-3 months')
    AND sl.products_sold > 30;

-- requête 4
WITH max_order_date AS (
    SELECT MAX(order_purchase_timestamp) AS max_date
    FROM orders
),
filtered_reviews AS (
    SELECT 
        r.review_score,
        c.customer_zip_code_prefix
    FROM 
        order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN max_order_date m ON 1=1
    WHERE 
        o.order_purchase_timestamp >= DATE(m.max_date, '-12 months')
),
reviews_aggregated AS (
    SELECT 
        customer_zip_code_prefix,
        COUNT(*) AS review_count,
        AVG(review_score) AS avg_review_score
    FROM 
        filtered_reviews
    GROUP BY 
        customer_zip_code_prefix
    HAVING 
        review_count > 30
)
SELECT 
    customer_zip_code_prefix,
    review_count,
    avg_review_score
FROM 
    reviews_aggregated
ORDER BY 
    avg_review_score ASC
LIMIT 5;
