# Databricks notebook source
# Make sure that pytest-bdd is installed on the cluster. This can be done by adding pytest-bdd as PyPi library on the cluster

# COMMAND ----------

pytestini = """
[pytest]
filterwarnings =
    error
    ignore::DeprecationWarning
markers =
    add
    basket
    empty
    error
    full
    outline
    remove
    unit
"""


with open("pytest.ini", "w") as fw:
    fw.write(pytestini)

# COMMAND ----------

cucumber = """
class CucumberBasket:

    def __init__(self, initial_count=0, max_count=10):
        if initial_count < 0:
            raise ValueError("Initial cucumber basket count must not be negative")
        if max_count < 0:
            raise ValueError("Max cucumber basket count must not be negative")

        self._count = initial_count
        self._max_count = max_count

    @property
    def count(self):
        return self._count

    @property
    def full(self):
        return self.count == self.max_count

    @property
    def empty(self):
        return self.count == 0

    @property
    def max_count(self):
        return self._max_count

    def add(self, count=1):
        new_count = self.count + count
        if new_count > self.max_count:
            raise ValueError("Attempted to add too many cucumbers")
        self._count = new_count

    def remove(self, count=1):
        new_count = self.count - count
        if new_count < 0:
            raise ValueError("Attempted to remove too many cucumbers")
        self._count = new_count

"""

with open("cucumber.py", "w") as fw:
    fw.write(cucumber)

# COMMAND ----------

test_unit_basic = """

# Partial Step Helpers
from functools import partial
from pytest_bdd import scenarios, given, when, then, parsers
from cucumber import CucumberBasket 

CONVERTERS = {
    'initial': int,
    'some': int,
    'total': int,
}

given_basket = partial(given, target_fixture='basket', converters=CONVERTERS)
when_cukes = partial(when, converters=CONVERTERS)
then_cukes = partial(then, converters=CONVERTERS)


# Scenarios

scenarios('unit_basic.feature')


# Given Steps

@given_basket(parsers.re(r'the basket has "(?P<initial>\d+)" cucumber(s?)'))
def basket_init(initial):
    return CucumberBasket(initial_count=initial)


@given_basket('the basket is empty')
def basket_empty():
    return CucumberBasket()


@given_basket('the basket is full')
def basket_full():
    return CucumberBasket(initial_count=10)


# When Steps

@when_cukes(parsers.re(r'"(?P<some>\d+)"( more)? cucumber(s?) are added to the basket'))
def add_cucumbers(basket, some):
    basket.add(some)


@when_cukes(parsers.re(r'"(?P<some>\d+)"( more)? cucumber(s?) are removed from the basket'))
def remove_cucumbers(basket, some):
    basket.remove(some)


# Then Steps

@then_cukes(parsers.re(r'the basket contains "(?P<total>\d+)" cucumbers'))
def basket_has_total(basket, total):
    assert basket.count == total


@then_cukes('the basket is empty')
def basket_is_empty(basket):
    assert basket.empty


@then_cukes('the basket is full')
def basket_is_full(basket):
    assert basket.full


@then_cukes(parsers.re(r'"(?P<some>\d+)" cucumbers cannot be added to the basket'))
def cannot_add_more(basket, some):
    count = basket.count
    try:
        basket.add(some)
    except ValueError as e:
        assert str(e) == "Attempted to add too many cucumbers"
        assert count == basket.count, "Cucumber count changed despite overflow"
    except:
        assert False, "Exception raised for basket overflow was not a ValueError"
    else:
        assert False, "ValueError was not raised for basket overflow"


@then_cukes(parsers.re(r'"(?P<some>\d+)" cucumbers cannot be removed from the basket'))
def cannot_remove_more(basket, some):
    count = basket.count
    try:
        basket.remove(some)
    except ValueError as e:
        assert str(e) == "Attempted to remove too many cucumbers"
        assert count == basket.count, "Cucumber count changed despite overdraw"
    except:
        assert False, "Exception raised for basket overdraw was not a ValueError"
    else:
        assert False, "ValueError was not raised for basket overdraw"

"""


with open("test_unit_basic.py", "w") as fw:
    fw.write(test_unit_basic)

# COMMAND ----------

unit_basic = """
@unit @basket
Feature: Cucumber Basket
  As a gardener,
  I want to carry cucumbers in a basket,
  So that I don't drop them all.

  # Gherkin-based automation frameworks *can* be used for unit testing.
  # However, they are better suited for integration and end-to-end testing.
  # This feature file does unit testing for the sake of illustrating Gherkin usage.

  @add
  Scenario: Add cucumbers to a basket
    Given the basket has "2" cucumbers
    When "4" cucumbers are added to the basket
    Then the basket contains "6" cucumbers

  @add @full
  Scenario: Fill the basket with cucumbers
    Given the basket is empty
    When "10" cucumbers are added to the basket
    Then the basket is full

  @add @error
  Scenario: Overfill the basket with cucumbers
    Given the basket has "8" cucumbers
    Then "3" cucumbers cannot be added to the basket

  @remove
  Scenario: Remove cucumbers from the basket
    Given the basket has "8" cucumbers
    When "3" cucumbers are removed from the basket
    Then the basket contains "5" cucumbers

  @remove @empty
  Scenario: Empty the basket of all cucumbers
    Given the basket is full
    When "10" cucumbers are removed from the basket
    Then the basket is empty

  @remove @error
  Scenario: Remove too many cucumbers from the basket
    Given the basket has "1" cucumber
    Then "2" cucumbers cannot be removed from the basket

  @add @remove
  Scenario: Add and remove cucumbers
    Given the basket is empty
    When "4" cucumbers are added to the basket
    And "6" more cucumbers are added to the basket
    But "3" cucumbers are removed from the basket
    Then the basket contains "7" cucumbers

"""


with open("unit_basic.feature", "w") as fw:
    fw.write(unit_basic)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l
# MAGIC python -m pytest test_unit_basic.py

# COMMAND ----------


