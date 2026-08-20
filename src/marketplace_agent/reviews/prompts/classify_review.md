Классифицируй один отзыв о ноутбуке только по фиксированной таксономии.

Отзыв:
{review}

Таксономия:
{taxonomy}

Правила:

- `product_defect`: дефект относится к товару; выбери известный `defect_id`
  или `other`, укажи `severity` (`low`, `medium` или `high`).
- `delivery_or_service`: отзыв только о доставке, курьере или обслуживании;
  верни `defect_id="no_defect"` и `severity=null`.
- `no_defect`: в отзыве нет дефекта товара; верни
  `defect_id="no_defect"` и `severity=null`.
- Не придумывай новые ID.
- `confidence` — число от 0 до 1.

Верни только JSON с полями `defect_id`, `kind`, `severity`, `confidence`.