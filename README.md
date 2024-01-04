# michimass

Interpolation and surface fitting

Interpolation auf unregelmäßigem Gitter

## Problemstellung

Gegeben die x-y Koordinaten und Messwerte an n Messpunkten `(xi, yi, zi)` für `i = 1..n`
und ein Koordinatenpunkt `(x, y)`.

Gesucht Schätzwert für erwarteten Messwert an der Stelle `f(x, y)`, so dass die Funktion `f`
an den Stellen `(xi, yi)` interpoliert, d.h. `f(xi, yi) = zi`, und `f` möglichst „glatt“ ist.

Ansatz `f(x’, y’) = a + b * x’ + c * y’` für alle `(x’, y’)` nahe bei `(x, y)`.
Die parameter `(a, b, c)` beschreiben eine Ebene im `(x, y, z)` – Raum.

Für jedes `(x, y)` werden andere `(a, b, c)` berechnet, so dass `Σ(((f(xi, yi) – zi) * wi)^2)`
möglichst klein wird (Gauss’sche Ausgleichsrechnung).

Dabei sind die `wi > 0` Gewichtsfaktoren, die stetig von `(x, y)` abhängen und die dafür sorgen, dass
in der Nähe der Messpunkte die Interpolationseigenschaft erfüllt wird.

Beispielsweise kann man `wi(x,y) = (dist((x,y), (xi,yi)) + eps) ^ -p` nehmen.
Dabei ist `eps` eine kleine Zahl, die dafür sorgt, dass an den Interpolationsstellen nicht Unendlich auftritt,
sondern eine große Zahl. Der Parameter `p` ( zwischen 2 und 3 ) steuert das Aussehen der Funktion.

Alternativ wäre auch denkbar `wi(x, y) = exp((1 – (dist + eps) / (mindist + eps)) * p)`,
wobei `mindist = minimum(dist((x,y), (xi,yi)) i = 1..n)` und `p` ein Steuerparameter ist.

## Ausgleichsrechnung - Berechnung von (a, b, c)

Die Ausdrücke in der summenformel schreibt man getrennt für jedes i:

```doc
    1 * a + x_1 * b + y_1 * c ~ z_1
    1 * a + x_2 * b + y_2 * c ~ z_ x_2
        …
    1 * a + x_n * b + y_n * c ~ z_n
```

oder in Matrix-Schreibweise

`A *   abc   ~   z` mit

```doc
                 1  x_1  y_1             z_1             a
mit ```  A =     1  x_2  y_2     und z = z_2     abc =   b
                        …                 …              c
                 1  x_n  y_n             z_n
```

Mit der diagonalen Gewichtsmatrix `W = diag(w)` dann

`D * A * abc  ~ D * Z`

mit der optimalen Lösung (d.h. die Summe wird minimiert)

`abc = pinv(D * A) * (D * z)` ; `pinv` ist die
[Moore-Penrose-Pseudo-Inverse](https://de.wikipedia.org/wiki/Pseudoinverse)
von `D*A`.
