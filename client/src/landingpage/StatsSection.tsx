import React, { useState, useEffect, useRef } from "react";
import { motion, useInView } from "framer-motion";
import { Brain, Users, Clock, Layers } from "lucide-react";

interface StatItem {
  icon: React.ElementType;
  value: number;
  suffix: string;
  label: string;
  color: string;
}

const stats: StatItem[] = [
  {
    icon: Brain,
    value: 10000,
    suffix: "+",
    label: "Models Trained",
    color: "from-slate-900 to-slate-700",
  },
  {
    icon: Users,
    value: 500,
    suffix: "+",
    label: "Active Users",
    color: "from-emerald-500 to-emerald-600",
  },
  {
    icon: Clock,
    value: 99,
    suffix: ".9%",
    label: "Uptime",
    color: "from-blue-500 to-blue-600",
  },
  {
    icon: Layers,
    value: 50,
    suffix: "+",
    label: "Templates",
    color: "from-violet-500 to-violet-600",
  },
];

const CountUp: React.FC<{ target: number; suffix: string }> = ({
  target,
  suffix,
}) => {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (!isInView) return;

    let start = 0;
    const duration = 2000;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);

    return () => clearInterval(timer);
  }, [isInView, target]);

  return (
    <span ref={ref}>
      {count.toLocaleString()}
      {suffix}
    </span>
  );
};

const StatsSection: React.FC = () => {
  return (
    <section className="relative py-24 bg-slate-50 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#e5e7eb_1px,transparent_1px),linear-gradient(to_bottom,#e5e7eb_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl lg:text-4xl font-bold bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
            Built for Scale
          </h2>
          <p className="mt-3 text-lg text-slate-600 font-light">
            Numbers that speak for themselves
          </p>
        </motion.div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="group text-center bg-white/80 backdrop-blur-xl p-8 rounded-2xl border border-slate-200/60 shadow-lg shadow-slate-900/5 hover:shadow-xl hover:shadow-slate-900/10 transition-all duration-300 ring-1 ring-slate-900/5"
              >
                <div
                  className={`w-14 h-14 mx-auto rounded-xl bg-linear-to-br ${stat.color} flex items-center justify-center shadow-lg mb-5 group-hover:scale-110 transition-transform`}
                >
                  <Icon className="w-7 h-7 text-white" />
                </div>
                <p className="text-4xl lg:text-5xl font-bold bg-linear-to-br from-slate-900 to-slate-700 bg-clip-text text-transparent">
                  <CountUp target={stat.value} suffix={stat.suffix} />
                </p>
                <p className="mt-2 text-sm font-medium text-slate-500 uppercase tracking-wide">
                  {stat.label}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default StatsSection;
