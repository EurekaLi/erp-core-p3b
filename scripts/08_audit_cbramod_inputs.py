"""只读原epochs，审计真实EEG的CBraMod适配；不产生正式分类成绩或模型权重。"""

from erp_core_p3b.cbramod_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
