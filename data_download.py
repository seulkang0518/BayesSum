# import json
# import requests
# from bs4 import BeautifulSoup

# URL = "https://www.stat.cmu.edu/COM-Poisson/Sales-data.html"

# def fetch_html(url):
#     r = requests.get(url, timeout=20)
#     # the page has Windows smart quotes etc.; try cp1252 then fallback
#     try:
#         r.encoding = "cp1252"
#         html = r.text
#     except Exception:
#         r.encoding = r.apparent_encoding or "utf-8"
#         html = r.text
#     return html

# def parse_two_col_int_table(html):
#     soup = BeautifulSoup(html, "html.parser")
#     for table in soup.find_all("table"):
#         rows = []
#         for tr in table.find_all("tr"):
#             cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
#             if len(cells) == 2:
#                 rows.append(cells)

#         # Try to coerce every row to ints (skip header-like rows)
#         pairs = []
#         for a, b in rows:
#             try:
#                 ai = int(a)
#                 bi = int(b)
#                 # skip obviously invalid (negative) lines
#                 if ai >= 0 and bi >= 0:
#                     pairs.append((ai, bi))
#             except ValueError:
#                 # not numeric (likely header), just skip
#                 continue

#         if pairs:
#             return dict(pairs)

#     return None

# def main():
#     html = fetch_html(URL)
#     sales_hist = parse_two_col_int_table(html)

#     # Final fallback: known histogram from the page (use only if scraping fails)
#     if not sales_hist:
#         sales_hist = {0: 514, 1: 503, 2: 276, 3: 148, 4: 65, 5: 24, 6: 9, 7: 5, 8: 2, 9: 1, 30: 1}

#     print("sales_hist =", sales_hist)

#     # Save to CSV and JSON for convenience
#     try:
#         import csv
#         with open("sales_hist.csv", "w", newline="") as f:
#             w = csv.writer(f)
#             w.writerow(["count", "freq"])
#             for k in sorted(sales_hist):
#                 w.writerow([k, sales_hist[k]])
#     except Exception as e:
#         print("CSV save failed:", e)

#     try:
#         with open("sales_hist.json", "w") as f:
#             json.dump(sales_hist, f)
#     except Exception as e:
#         print("JSON save failed:", e)

# if __name__ == "__main__":
#     main()

import numpy as np

hist = {0: 514, 1: 503, 2: 457, 3: 423, 4: 326, 5: 233, 6: 195, 7: 139, 8: 101,
        9: 77, 10: 56, 11: 40, 12: 37, 13: 22, 14: 9, 15: 7, 16: 10, 17: 9,
        18: 3, 19: 2, 20: 2, 21: 2, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0,
        28: 0, 29: 0, 30: 1}

xs = np.array(sorted(hist.keys()))
fs = np.array([hist[k] for k in xs])
n  = fs.sum()
mean = (xs*fs).sum()/n
var  = (xs**2*fs).sum()/n - mean**2
disp = var/mean
p0   = hist.get(0,0)/n
p0_pois = np.exp(-mean)

nu0 = mean/var
lam0 = (mean + (nu0-1)/(2*nu0))**nu0

print(n, mean, var, disp, p0, p0_pois, nu0, lam0)