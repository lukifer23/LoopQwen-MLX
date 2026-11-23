from fractions import Fraction
from decimal import Decimal
from math import comb

CATEGORIES=("integer","multi_step","decimal","fraction","percent","precedence","word","equation","count","sequence","algorithm")


def problem(rng, category):
    a,b=sorted(rng.sample(range(2,180),2))
    c=rng.randint(2,19)
    if category=="integer":
        expression,answer,key,steps=f"{a} + {b}",a+b,(a,b),1
    elif category=="multi_step":
        expression,answer,key,steps=f"({b} - {a}) × {c}",(b-a)*c,(a,b,c),2
    elif category=="decimal":
        expression=f"{Decimal(a)/10} + {Decimal(b)/100}"
        answer,key,steps=Decimal(a)/10+Decimal(b)/100,(a,b),1
    elif category=="fraction":
        expression,answer,key,steps=f"{a}/{c} + {b}/{c}",Fraction(a+b,c),(a,b,c),2
    elif category=="percent":
        expression,answer,key,steps=f"{a}% of {b}",Decimal(a*b)/100,(a,b),2
    elif category=="precedence":
        expression,answer,key,steps=f"{a} + {b} × {c}",a+b*c,(a,b,c),2
    elif category=="word":
        expression=f"A box holds {b} beads. I remove {a}, then add {c}. How many beads are in the box?"
        answer,key,steps=b-a+c,(a,b,c),2
    elif category=="equation":
        expression=f"Solve for x: {c}x + {a} = {c*b+a}"
        answer,key,steps=b,(a,b,c),2
    elif category=="count":
        expression=f"How many different pairs can be chosen from {a+b} people?"
        # Include a second genuine degree of freedom for ample distinct instances.
        count=2+(c%3)
        expression=f"How many groups of {count} can be chosen from {a+b} people?"
        answer,key,steps=comb(a+b,count),(a+b,count),count
    elif category=="sequence":
        expression=f"Continue the arithmetic sequence: {a}, {a+c}, {a+2*c}, ..."
        answer,key,steps=a+3*c,(a,c),2
    else:
        expression=f"What is the remainder when {b*c+a} is divided by {c}?"
        answer,key,steps=(b*c+a)%c,(b*c+a,c),2
    prompt=rng.choice(("Question: {}\nAnswer:","Problem: {}\nAnswer:","{}\nAnswer:")).format(expression)
    return dict(id=category+"-"+"-".join(map(str,key)),category=category,steps=steps,prompt=prompt,answer=str(answer))
