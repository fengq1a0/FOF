#include <cstdio>
#include <cmath>
#include <limits>
#include <algorithm>
#include <ctime>

using namespace std;

extern "C"
{
    void welzl(double* v, int n, double* ans);
}

template <typename T>
T sq(T x) {
    return x * x;
}

const double eps = 1e-12, pi = acos(-1.0);
int sgn(double x) {
    return fabs(x) < eps ? 0 : (x < 0 ? -1 : 1);
}
struct VEC {
    double x, y, z;

    VEC operator + (const VEC& b) const {
        return VEC{x + b.x, y + b.y, z + b.z};
    }
    VEC operator - (const VEC& b) const {
        return VEC{x - b.x, y - b.y, z - b.z};
    }

    double len2() const {
        return sq(x) + sq(y) + sq(z);
    }
    double len() const {
        return sqrt(len2());
    }

    double dist2(const VEC& b) const {
        return (*this - b).len2();
    }
    double dist(const VEC& b) const {
        return (*this - b).len();
    }

    VEC operator * (double b) const {
        return VEC{x * b, y * b, z * b};
    }
    VEC trunc(double l) const {
        double ori = len();
        if (!sgn(ori)) return *this;
        return *this * (l / ori);
    }

    VEC operator / (double b) const {
        return VEC{x / b, y / b, z / b};
    }
    VEC norm() const {
        return *this / len();
    }

    VEC operator ^ (const VEC& b) const {
        return VEC{y * b.z - b.y * z, b.x * z - x * b.z, x * b.y - b.x * y};
    }
    double operator * (const VEC& b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    double rad(const VEC& b) const {
        return fabs(atan2((*this ^ b).len(), *this * b));
    }
    double rad_acos(const VEC& b) const {
        double L1 = len2(), L2 = b.len2();
        return acos((L1 + L2 - (*this - b).len2()) / (2 * sqrt(L1) * sqrt(L2)));
    }

    double MixedProd(const VEC& b, const VEC& c) const {
        return *this * (b ^ c);
    }
};

struct LINE {
    VEC s, v;
    double dist(const VEC& p) const {
        return (v ^ (p - s)).len() / v.len();
    }
    VEC projection(const VEC& p) const {
        VEC v1 = v.norm();
        return v1 * ((p - s) * v1) + s;
    }
};

struct CIRCLE3 {
    VEC c, v;
    double r;
};

struct BALL {
    VEC c;
    double r;

    bool operator == (const BALL& b) const {
        return sgn((c - b.c).len()) == 0 && sgn(r - b.r) == 0;
    }
    //1: inside
    //2: internally tangent
    //3: Cross
    //4: externally tangent
    //5: away from
    int relation(const BALL& b) const {
        double d = c.dist(b.c);
        int s = sgn(d - fabs(b.r - r));
        if (s <= 0) return s + 2;
        return sgn(d - (b.r + r)) + 4;
    }

    //-1: inside
    //0: on
    //1: out
    int relation(const VEC& p) const {
        return sgn(p.dist(c) - r);
    }

    //-1: infinite
    int cross(CIRCLE3& ans, const BALL& b) const {
        int rel = relation(b);
        if (rel == 1 || rel == 5) return 0;
        if (*this == b) return -1;
        ans.v = (b.c - c).norm();
        double d = c.dist(b.c);
        double l = (sq(r) + sq(d) - sq(b.r)) / (2 * d);
        ans.c = c + ans.v * l;
        ans.r = sqrt(sq(r) - sq(l));
        return 1;
    }

    void SetCircumscribed(const VEC& A, const VEC& B) {
        c = (A + B) / 2.0;
        r = (B - A).len() / 2.0;
    }

    //outer
    //Make sure that A, B, C are not collinear
    void SetCircumscribed(const VEC& A, VEC B, VEC C) {
        B = B - A;
        C = C - A;
        VEC s2di = B ^ C;
        if (s2di.len() < eps) return;
        VEC abv = s2di ^ B;
        VEC acv = C ^ s2di;
        VEC to = (abv * C.len2() + acv * B.len2()) / (2.0 * s2di.len2());
        c = A + to;
        r = to.len();
    }
    //Make sure that A, B, C, D are not in the same plane
    void SetCircumscribed(const VEC& A, const VEC& B, const VEC& C, VEC D) {
        SetCircumscribed(A, B, C);
        VEC v = ((C - A) ^ (B - A)).norm();
        D = D - c;
        double L = D * v, d2 = (v ^ D).len2();
        if (fabs(L) < eps) return;
        double k = (d2 + sq(L) - sq(r)) / (2 * L);
        c = c + v * k;
        r = sqrt(sq(k) + sq(r));
    }
};

void welzl(double* v, int n, double* ans) {
    VEC* ps = (VEC*) v;
    random_shuffle(ps, ps + n);
    BALL ball{ps[0], 0};
    for (int i1 = 1; i1 < n; ++i1) {
        if (ball.relation(ps[i1]) == 1) {
            ball.c = ps[i1];
            for (int i2 = 0; i2 < i1; ++i2) {
                if (ball.relation(ps[i2]) == 1) {
                    ball.SetCircumscribed(ps[i1], ps[i2]);
                    for (int i3 = 0; i3 < i2; ++i3) {
                        if (ball.relation(ps[i3]) == 1) {
                            ball.SetCircumscribed(ps[i1], ps[i2], ps[i3]);
                            for (int i4 = 0; i4 < i3; ++i4) {
                                if (ball.relation(ps[i4]) == 1) {
                                    ball.SetCircumscribed(ps[i1], ps[i2], ps[i3], ps[i4]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ans[0] = ball.c.x;
    ans[1] = ball.c.y;
    ans[2] = ball.c.z;
    ans[3] = ball.r;
}

#define MAXN 111
int main() {
    int n;
    VEC ps[MAXN];

    srand(19990828);

    while (~scanf("%d", &n) && n) {
        for (int i = 0; i < n; ++i) {
            int tmp = scanf("%lf%lf%lf", &ps[i].x, &ps[i].y, &ps[i].z);
        }
        double ans[4];
        welzl((double*)ps, n, ans);
        printf("%.5f\n", ans[3]);
    }

    return 0;
}
