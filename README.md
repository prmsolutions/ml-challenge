# Proton Data Science Challenge v1

Congratulations, you've passed our technical interview! Now it's time for a more in-depth chance to get a sense of the problems we solve and how we solve them.

We've built a small model here for predicting items people should add to their shopping basksets. We'll give you instructions separately on how we'd like you to improve it.


# FAQ

## 1. What does the function `internal_utils.get_product_count(client_name)` return?
This function will return an integer representing the total number of products that a specific client(distributor) carries in their product catalog.\
For example, say Bob's Distribution Company works with restaurants and carries three products:
Olive Oil, Tomato Sauce and Pizza dough.\
In this case `internal_utils.get_product_count("bobs_distribution_company")` will return the integer 3.\
For the purposes of the challenge you can assume the number will always be between 100k - 500k and that this number will be different for each client

## 2. Client and customer. What means what?
Great question! This can sometimes be a source of confusion especially when those terms show up together very often.\
When we say **client** we mean one of our clients, i.e. a distributor. When we say **customer** we refer to one of our 
client's (distributor's) customers. 

## 3. Can you better explain what do the order carts look like?
To step back a bit, an order cart is a bundle of products bought by one of our **client's** customers. E.g. Daniel's Pizzeria ordered a bundle of products
from one of our clients. That bundle will be one of the order carts. 

Here's an example of an order cart. Say Daniel's Pizzeria buys two products from one of **our** clients called Jon The Distributor.
Those two products are 2 jars of "Anna's Tomato Sauce" and 4 pies of "Bob's Pizza Dough". Here's what that cart might look like
within the context of the problem.

```
    {
        "order_id": "1Zef45GH4689990345gHtR",
        "customer_id": "12-daniels_pizzeria-H",
        "order_date": "2021-02-11T18:46:11.785341",
        "products": [
            {
                "product_name": "Anna's Tomato Sauce", 
                "product_id": "anna_tomato_sauce_004",
                "product_category": "sauces",
                "product_description": "Old fashioned tomato sauce as made by Grandma Anna back in the day, using all-natural ingredients. 20 oz jar.",
                "price": 6.99,
                "manufacturer": "Anna's Famous Sauces",
                "quantity_purchased": 2,
                "purchase_date": '2021-02-11T18:46:11.785341',
                "product_index": 356,
            },
            {
                "product_name": "Bob's Pizza Dough",
                "product_id": "bob_pizza_dough_XT3",
                "product_category": "baked_goods",
                "product_description": "Pizza dough made following time-tested traditions. Great for pizzas and calzones. One 16 inch pie.",
                "price": 10.99,
                "manufacturer": "Bob's Baking Goods",
                "quantity_purchased": 4,
                "purchase_date": '2021-02-11T18:46:11.785341',
                "product_index": 0,
            }
        ]
    }
```
Note that the descriptions are not restricted to being as short as the ones above (they are just an example).
Hopefully this makes visualizing things a little bit easier. 

## 4. Can you explain what the product index field is? I don't really get it.
Yes! Say we have a client called Jon The Distributor. They carry 4 products in total, which include "Anna's Tomato Sauce",
"Bob's Pizza Dough", "Premium Salami Slices" and "Fresh Mozzarella di Bufala".
Then "Anna's Tomato Sauce" would have index 0, "Bob's Pizza Dough" would have index 1, "Premium Salami Slices" index 2 and "Fresh Mozzarella di Bufala" an index of 3.

This index is set once and is never changed after. If a new product were to be added, say "Fresh Pecorino Romano Cheese" it
would get the index of 4 (3 + 1). 


## 5. I have a question that was no answered by the FAQ, what should I do?
In that case you should contact the person that was mentioned in the email sent to you along with the assignment itself and someone from
our data science team will get back to you!
