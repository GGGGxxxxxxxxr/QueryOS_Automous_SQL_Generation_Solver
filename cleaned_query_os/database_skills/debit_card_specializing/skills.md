# debit_card_specializing Skills

This DB combines customer profiles, customer-month consumption summaries, transaction records, gas station metadata, and product metadata. Always identify the entity grain before writing SQL.

## Grain Rules

- `yearmonth`: customer-month grain. `Consumption` is not a whole-month total.
- `transactions_1k`: transaction grain.
- `customers`: customer attributes such as segment and currency.
- `gasstations`: station attributes such as country and station segment.
- `products`: product attributes.

## Consumption

Month-level consumption requires aggregating customer-month rows:

```sql
GROUP BY <month_expr>
ORDER BY SUM(<consumption>) DESC
```

Customer-year consumption requires aggregating by customer:

```sql
GROUP BY <customer_id>
```

Do not use `MAX(<consumption>)` for a peak month unless the question asks for one customer's maximum monthly value.

## Counting

Default to row/event counting. Do not use `DISTINCT` unless the question explicitly asks for unique customers, distinct entities, or customer identities.

```sql
COUNT(*)                       -- records, monthly rows, transactions
COUNT(DISTINCT <customer_id>)  -- unique customers only
```

For percentages, numerator and denominator must use the same grain.

## Entity Mapping

- Customer segment and gas station segment are different.
- Customer currency and gas station country are different.
- Country/location questions usually need the gas station path.
- In this DB, customer nationality/country is treated as the country of the gas station where the customer made the relevant transaction.
- Product-description questions need the product path.
- Transaction-date/price/amount questions should start from transactions and join outward.

## Dates

Use table-specific date logic:

```sql
-- customer-month table
<date> BETWEEN '<YYYYMM>' AND '<YYYYMM>'
SUBSTR(<date>, 1, 4)
SUBSTR(<date>, 5, 2)

-- transaction table
<date> = '<YYYY-MM-DD>'
```

## Final Check

Before submission, verify:

1. target grain;
2. row count first; distinct customer count only if explicitly required;
3. consumption aggregated to the requested level;
4. segment/currency/country mapped to the right entity;
5. date format matches the table.
