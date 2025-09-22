import math
import random
from typing import Callable, Dict, Any

from taxmusr.core.schemas import Person, CoupleTaxInput

TaxFunc = Callable[[float], float]
IMBALANCED = [(58000, 0), (60000, 6000), (95000, 22000)]
SIMILAR    = [(72000, 70000), (40000, 42000), (55000, 53000)]


def compute_tax_2025(x: float) -> int:
    """
    Compute the tax for a given income x based on the provided brackets.
    Based on: https://www.gesetze-im-internet.de/estg/__32a.html for the year 2025.
    You can add your own implementation of the tax function if needed.
    :param x: the taxable income
    :return: the computed tax
    """
    e1, e2, e3, e4 = (12096, 17443, 68480, 277825)
    x = math.floor(max(0.0, x))

    if x <= e1:
        tax = 0.0
    elif x <= e2:
        # y is a ten-thousandth of the part exceeding the basic allowance
        y = (x - e1) / 10000.0
        tax = (932.30 * y + 1400.0) * y
    elif x <= e3:
        # z is a ten-thousandth of the part exceeding 17,443
        z = (x - e2) / 10000.0
        tax = (176.64 * z + 2397.0) * z + 1015.13
    elif x <= e4:
        tax = 0.42 * x - 10911.92
    else:
        tax = 0.45 * x - 19246.67

    return math.floor(tax)


def single_assessment(taxable_income: float, tax_function: TaxFunc=compute_tax_2025) -> float:
    """
    Compute the income tax for a single person based on the provided tax function.
    :param taxable_income: the taxable income
    :param tax_function: the tax function to use (default is marginal_tax_2025)
    :return: the computed tax
    """
    return tax_function(max(0.0, taxable_income))


def joint_assessment(taxable_income: float, tax_function: TaxFunc=compute_tax_2025) -> float:
    """
    Compute the income tax for a couple using the splitting method based on the provided tax function.
    :param taxable_income: the combined taxable income
    :param tax_function: the tax function to use (default is marginal_tax_2025)
    :return: the computed tax
    """
    half = max(0.0, taxable_income) / 2.0
    return 2.0 * tax_function(half)


def progression_rate_with_wrb(taxable_income: float, wage_replacement: float,
                              joint: bool=False, tax_function: TaxFunc=compute_tax_2025) -> float:
    """
    Compute the progression rate for wage replacement benefits.
    Wage replacement benefits are tax-free, but they increase the tax rate on the taxable income.
    :param taxable_income: the taxable income
    :param wage_replacement: the amount of wage replacement benefits
    :param joint: whether to use joint assessment (default is False, i.e., single assessment)
    :param tax_function: the tax function to use (default is marginal_tax_2025)
    :return:
    """
    base_plus = max(0.0, taxable_income) + max(0.0, wage_replacement)
    if base_plus <= 0:
        return 0.0
    if joint:
        tax_with_progression = joint_assessment(base_plus, tax_function)
    else:
        tax_with_progression = single_assessment(base_plus, tax_function)
    return tax_with_progression / base_plus


def get_taxable_income_after_medical(person: Person) -> float:
    """
    Compute the taxable income after deducting medical costs.
    Only the part above a certain threshold based on income is deductible.
    This is still a simplified model not taking into account all details (e.g. number of children, splitting tariff).
    Also, tax-free wage replacement benefits are not considered here.
    :param person: the person for whom to compute the taxable income
    :return: the taxable income after deducting medical costs
    """
    income = max(0.0, person.income)
    medical = max(0.0, person.medical_costs)

    # Thresholds for 2025
    if income <= 15340:
        threshold = 0.05 * income
    elif income <= 51130:
        threshold = 0.06 * income
    else:
        threshold = 0.07 * income

    deductible = max(0.0, medical - threshold)
    return max(0.0, income - deductible)


def compute_special_church_tax(income: float) -> float:
    """
    Compute the special church tax for couples where only one partner pays church tax and has no or
    significantly lower income (< 35% of the total income).
    https://www.kirchensteuer-wirkt.de/kirche-und-geld/finanzamt-kirchensteuer/besonderes-kirchgeld
    :param income: the total taxable income of the couple
    :return: the computed special church tax
    """
    if income < 50000:
        return 0.0
    elif income < 57500:
        return 96.0
    elif income < 70000:
        return 156.0
    elif income < 82500:
        return 276.0
    elif income < 95000:
        return 396.0
    elif income < 107500:
        return 540.0
    elif income < 120000:
        return 696.0
    elif income < 145000:
        return 840.0
    elif income < 170000:
        return 1200.0
    elif income < 195000:
        return 1560.0
    elif income < 220000:
        return 1860.0
    elif income < 270000:
        return 2220.0
    elif income < 320000:
        return 2940.0
    else:
        return 3600.0


# ------------------------------
# Core calculations
# ------------------------------

def compute_joint_total(params: CoupleTaxInput, tax_function: TaxFunc=compute_tax_2025) -> float:
    """
    Compute the total tax for a couple using joint assessment.
    :param params: the couple's tax input parameters
    :param tax_function: the tax function to use
    :return: the computed total tax
    """
    ta = get_taxable_income_after_medical(params.a)
    tb = get_taxable_income_after_medical(params.b)
    taxable_total = ta + tb

    wrb_total = max(0.0, params.a.wage_replacement) + max(0.0, params.b.wage_replacement)

    # Income tax under splitting with Progressionsvorbehalt
    prate = progression_rate_with_wrb(taxable_total, wrb_total, joint=True, tax_function=tax_function)
    income_tax_total = prate * taxable_total

    # Default: Allocate base tax proportionally for church tax
    base_total = income_tax_total
    share_a = ta / taxable_total if taxable_total > 0 else 0.0
    share_b = tb / taxable_total if taxable_total > 0 else 0.0

    alloc_a = base_total * share_a
    alloc_b = base_total * share_b

    church_a = alloc_a * params.church_tax_rate if params.a.pays_church_tax else 0.0
    church_b = alloc_b * params.church_tax_rate if params.b.pays_church_tax else 0.0

    # Special church tax applies if only one pays church tax, but has no income/ very low income
    # it is calculated on the full income tax
    if params.a.pays_church_tax != params.b.pays_church_tax:
        if params.a.pays_church_tax and share_a < 0.35:
            special_church_tax = compute_special_church_tax(taxable_total)
            church_a = max(church_a, special_church_tax)
        elif params.b.pays_church_tax and share_b < 0.35:
            special_church_tax = compute_special_church_tax(taxable_total)
            church_b = max(church_b, special_church_tax)

    return round(base_total + church_a + church_b, 2)


def compute_individual_total(params: CoupleTaxInput, tax_function: TaxFunc=compute_tax_2025) -> float:
    """
    Compute the total tax for individual assessment.
    :param params: the couple's tax input parameters
    :param tax_function: the tax function to use
    :return: the computed total tax
    """
    # Partner A
    ta = get_taxable_income_after_medical(params.a)
    prate_a = progression_rate_with_wrb(ta, params.a.wage_replacement, joint=False, tax_function=tax_function)
    base_a = prate_a * ta
    church_a = base_a * params.church_tax_rate if params.a.pays_church_tax else 0.0
    total_a = base_a + church_a

    # Partner B
    tb = get_taxable_income_after_medical(params.b)
    prate_b = progression_rate_with_wrb(tb, params.b.wage_replacement, joint=False, tax_function=tax_function)
    base_b = prate_b * tb
    church_b = base_b * params.church_tax_rate if params.b.pays_church_tax else 0.0
    total_b = base_b + church_b

    return round(total_a + total_b, 2)


def compare_assessments(params: CoupleTaxInput, tax_function: TaxFunc=compute_tax_2025) -> Dict[str, Any]:
    joint_total = compute_joint_total(params, tax_function)
    individual_total = compute_individual_total(params, tax_function)

    advantage = round(individual_total - joint_total, 2)  # positive -> joint saves that amount
    # recommend "joint" if tied
    recommendation = "joint" if joint_total < individual_total else ("individual" if individual_total < joint_total else "joint")

    return {
        "individual_total_tax": individual_total,
        "joint_total_tax": joint_total,
        "advantage_if_positive_joint_saves": advantage,
        "recommendation": recommendation
    }


def sample_couple_input() -> CoupleTaxInput:
    if random.random() < 0.5:
        income_a, income_b = random.choice(IMBALANCED)
    else:
        income_a, income_b = random.choice(SIMILAR)
    # add noise
    income_a = max(0, int(random.gauss(income_a, 5000)))
    income_b = max(0, int(random.gauss(income_b, 5000)))

    pays_church_a = random.random() < 0.3
    pays_church_b = random.random() < 0.3
    wage_replacement_a = random.choice([0, 10800, 21600])
    wage_replacement_b = 0
    if random.random() < 0.3:
        medical_costs_a = random.choice([500, 2000, 5000])
        medical_costs_a = max(0, int(random.gauss(medical_costs_a, 300)))
    else:
        medical_costs_a = 0
    if random.random() < 0.3:
        medical_costs_b = random.choice([500, 2000, 5000])
        medical_costs_b = max(0, int(random.gauss(medical_costs_b, 300)))
    else:
        medical_costs_b = 0
    church_tax_rate = 0.09 if random.random() < 0.8 else 0.08   # Bavaria and Baden-Württemberg have 8%
    person_a = Person(
        income=income_a, pays_church_tax=pays_church_a,
        wage_replacement=wage_replacement_a, medical_costs=medical_costs_a
    )
    person_b = Person(
        income=income_b, pays_church_tax=pays_church_b,
        wage_replacement=wage_replacement_b, medical_costs=medical_costs_b
    )
    live_together = random.choices([True, False], weights=[0.9, 0.1])[0]
    number_of_children = random.choices([0, 1, 2, 3], weights=[0.20, 0.24, 0.38, 0.18])[0]
    return CoupleTaxInput(
        a=person_a, b=person_b, church_tax_rate=church_tax_rate, children=number_of_children,live_together=live_together
    )

