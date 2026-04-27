import { useEffect, useRef, useState } from "react";

const defaultBeforeData = {
  score: 0.67,
  demographicParityDifference: 0.67,
  equalizedOddsDifference: 0.41,
  status: "Moderate Bias",
  badge: "Moderate",
  performance: {
    accuracy: 0.81,
    precision: 0.77,
    recall: 0.72,
  },
  groupMetrics: {
    Male: 72,
    Female: 42,
  },
  errorHeadline: "Model rejects females more often",
  errorText:
    "False negative rate is 1.7x higher for female applicants compared with male applicants.",
  insights: [
    "Gender is influencing predictions indirectly through correlated features.",
    "Female applicants have a 30% lower approval rate in the current model.",
    "Age contributes mild imbalance but is not the dominant driver.",
  ],
  alerts: [
    "Potential discrimination detected in approval outcomes.",
    "High bias signal linked to feature: Income.",
    "Protected group error rates exceed internal threshold.",
  ],
  recommendedAction:
    "Combine rebalancing with a fairness constraint to reduce the approval gap while preserving model quality.",
  recommendationSource: "demo",
};

const navItems = [
  { id: "landing", label: "Landing" },
  { id: "upload", label: "Upload Data" },
  { id: "loading", label: "Analyze" },
  { id: "dashboard", label: "View Report" },
  { id: "fix", label: "Fix Bias" },
  { id: "compare", label: "Compare Results" },
];

const defaultAfterData = {
  score: 0.21,
  demographicParityDifference: 0.21,
  equalizedOddsDifference: 0.17,
  performance: {
    accuracy: 0.79,
    precision: 0.76,
    recall: 0.75,
  },
  groupMetrics: {
    Male: 69,
    Female: 63,
  },
  recommendedAction:
    "Validate the mitigated model on a holdout split and confirm that the fairness lift holds across fresh samples.",
  recommendationSource: "demo",
};

function App() {
  const [currentScreen, setCurrentScreen] = useState("landing");
  const [selectedFile, setSelectedFile] = useState("No file selected");
  const [datasetFile, setDatasetFile] = useState(null);
  const [availableColumns, setAvailableColumns] = useState([]);
  const [availableSensitiveOptions, setAvailableSensitiveOptions] = useState(
    [],
  );
  const [columnSamples, setColumnSamples] = useState({});
  const [targetColumn, setTargetColumn] = useState("");
  const [sensitiveFeatures, setSensitiveFeatures] = useState([]);
  const [analysisData, setAnalysisData] = useState(defaultBeforeData);
  const [afterData, setAfterData] = useState(defaultAfterData);
  const [analysisMeta, setAnalysisMeta] = useState({
    datasetName: "",
    targetColumn: "",
    targetPositiveLabel: "",
    detectedSensitiveFeatures: [],
    targetNote: "",
    fairnessMetric: "Demographic Parity Difference",
  });
  const [apiError, setApiError] = useState("");
  const [requestMessage, setRequestMessage] = useState("");
  const [insightsSource, setInsightsSource] = useState("rule-based");
  const [insightsError, setInsightsError] = useState("");
  const [recommendationError, setRecommendationError] = useState("");
  const [hasMitigated, setHasMitigated] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isApplyingFixes, setIsApplyingFixes] = useState(false);
  const [fixes, setFixes] = useState({
    sensitive: true,
    rebalance: true,
    constraint: false,
  });

  const refs = {
    landing: useRef(null),
    upload: useRef(null),
    loading: useRef(null),
    dashboard: useRef(null),
    fix: useRef(null),
    compare: useRef(null),
  };

  useEffect(() => {
    refs[currentScreen]?.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, [currentScreen]);

  async function readCsvPreview(file) {
    const text = await file.text();
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    const firstLine = lines[0] || "";
    const headers = firstLine
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);

    const samples = {};
    headers.forEach((header) => {
      samples[header] = [];
    });

    lines.slice(1, 31).forEach((line) => {
      const cells = line.split(",").map((value) => value.trim());
      headers.forEach((header, index) => {
        const value = cells[index];
        if (value) {
          samples[header].push(value);
        }
      });
    });

    return { headers, samples };
  }

  function isBinaryLike(values) {
    if (!values || values.length === 0) {
      return false;
    }

    const normalized = values.map((value) =>
      String(value).trim().toLowerCase(),
    );
    const unique = new Set(normalized);
    return unique.size === 2;
  }

  function isContinuousFeature(header, samples, threshold = 6) {
    const values = samples[header] || [];
    if (values.length === 0) {
      return false;
    }

    const normalized = values
      .map((value) => String(value).trim().toLowerCase())
      .filter(Boolean);
    const uniqueCount = new Set(normalized).size;
    const numericCount = normalized.filter(
      (value) => !Number.isNaN(Number(value)),
    ).length;
    return (
      normalized.length > 0 &&
      numericCount / normalized.length >= 0.8 &&
      uniqueCount > threshold
    );
  }

  function suggestTargetColumn(headers, samples) {
    const byName = headers.find((header) =>
      /loan|approved|hired|decision|target|label|outcome|admit|accept|status|income/i.test(
        header,
      ),
    );
    if (byName) {
      return byName;
    }

    const binaryCandidate = headers.find((header) =>
      isBinaryLike(samples[header] || []),
    );
    return binaryCandidate || "";
  }

  function suggestSensitiveFeatures(headers, target, samples) {
    const candidates = headers.filter((header) => header !== target);
    const prioritized = candidates.filter((header) =>
      /sex|gender|age|race|ethnicity|region|country|state|marital|disability|religion|caste/i.test(
        header,
      ),
    );
    const remaining = candidates.filter(
      (header) => !prioritized.includes(header),
    );
    const safeRemaining = remaining.filter(
      (header) => !isContinuousFeature(header, samples, 10),
    );
    return [...prioritized, ...safeRemaining].slice(0, 4);
  }

  function buildSensitiveOptions(headers, target) {
    return headers.filter((header) => header !== target);
  }

  function getGroupEntries(groupMetrics) {
    return Object.entries(groupMetrics || {}).slice(0, 4);
  }

  function getPrimaryGroups(groupMetrics) {
    const entries = getGroupEntries(groupMetrics);
    if (entries.length === 0) {
      return [
        ["Group A", 0],
        ["Group B", 0],
      ];
    }
    if (entries.length === 1) {
      return [entries[0], entries[0]];
    }
    return entries.slice(0, 2);
  }

  function getComparisonGroups(beforeMetrics, afterMetrics) {
    const labels = Array.from(
      new Set([
        ...Object.keys(beforeMetrics || {}),
        ...Object.keys(afterMetrics || {}),
      ]),
    );

    return labels.slice(0, 6).map((label) => ({
      label,
      before: typeof beforeMetrics?.[label] === "number" ? beforeMetrics[label] : 0,
      after: typeof afterMetrics?.[label] === "number" ? afterMetrics[label] : 0,
    }));
  }

  async function parseApiResponse(response) {
    const raw = await response.text();

    try {
      return raw ? JSON.parse(raw) : {};
    } catch {
      if (!response.ok) {
        throw new Error(raw || "Server returned an invalid response.");
      }
      throw new Error("Server returned invalid JSON.");
    }
  }

  function toNumberOrNull(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatMetric(value, digits = 3) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(digits)
      : "N/A";
  }

  function formatPercent(value) {
    return typeof value === "number" && Number.isFinite(value)
      ? `${value}%`
      : "N/A";
  }

  function formatGroupMetrics(groupMetrics) {
    const entries = Object.entries(groupMetrics || {});
    if (entries.length === 0) {
      return "No group metrics available.";
    }

    return entries.map(([label, value]) => `- ${label}: ${formatPercent(value)}`).join("\n");
  }

  function buildReportContent() {
    const generatedAt = new Date().toLocaleString();
    const datasetName = analysisMeta.datasetName || "Sample demo dataset";
    const target = analysisMeta.targetColumn || "Unknown target";
    const readableOutcomeLabel = getReadableOutcomeLabel(
      analysisMeta.targetColumn,
      analysisMeta.targetPositiveLabel,
    );
    const sensitive = selectedSensitiveFeatures.length > 0
      ? selectedSensitiveFeatures.join(", ")
      : "No sensitive features selected";
    const auditScope = isIntersectional
      ? "Intersectional multi-group audit enabled"
      : "Single-attribute group audit enabled";
    const beforeSummary = [
      `Bias score: ${analysisData.score.toFixed(2)}`,
      `Severity: ${analysisData.status}`,
      `Primary fairness metric: ${analysisMeta.fairnessMetric}`,
      `Demographic parity difference: ${formatMetric(analysisData.demographicParityDifference)}`,
      `Equalized odds difference: ${formatMetric(analysisData.equalizedOddsDifference)}`,
      `Accuracy: ${formatMetric(analysisData.performance?.accuracy)}`,
      `Precision: ${formatMetric(analysisData.performance?.precision)}`,
      `Recall: ${formatMetric(analysisData.performance?.recall)}`,
    ].join("\n");

    const executiveSummary = hasMitigated
      ? `FairLens analyzed ${datasetName} and reduced the measured bias score from ${analysisData.score.toFixed(2)} to ${afterData.score.toFixed(2)} while changing accuracy by ${accuracyDelta}. The strongest current recommendation is: ${afterData.recommendedAction || analysisData.recommendedAction}.`
      : `FairLens analyzed ${datasetName} and found a ${analysisData.status.toLowerCase()} fairness risk with a bias score of ${analysisData.score.toFixed(2)}. The strongest current recommendation is: ${analysisData.recommendedAction}.`;
    const projectSummary = hasMitigated
      ? `FairLens completed a multi-group fairness review, highlighted the most affected populations, and evaluated mitigation impact across bias and performance metrics. The mitigated model now shows a fairness lift of ${fairnessLiftLabel} with an accuracy delta of ${accuracyDeltaLabel}.`
      : "FairLens completed a multi-group fairness review, highlighted the most affected populations, and generated a mitigation-ready decision summary for stakeholders.";

    const mitigationSection = hasMitigated
      ? `## Mitigation Results

Applied fixes:
- Remove sensitive feature: ${fixes.sensitive ? "Yes" : "No"}
- Rebalance dataset: ${fixes.rebalance ? "Yes" : "No"}
- Apply fairness constraint: ${fixes.constraint ? "Yes" : "No"}

After mitigation:
- Bias score: ${afterData.score.toFixed(2)}
- Demographic parity difference: ${formatMetric(afterData.demographicParityDifference)}
- Equalized odds difference: ${formatMetric(afterData.equalizedOddsDifference)}
- Accuracy: ${formatMetric(afterData.performance?.accuracy)}
- Precision: ${formatMetric(afterData.performance?.precision)}
- Recall: ${formatMetric(afterData.performance?.recall)}
- Fairness lift: ${Math.abs(fairnessLift)}%
- Accuracy delta: ${accuracyDelta}

Group outcome rates after mitigation:
${formatGroupMetrics(afterData.groupMetrics)}

Recommended next step:
${afterData.recommendedAction || "Review the mitigated model on fresh data before deployment."}
`
      : `## Mitigation Status

No mitigation has been applied yet. Recommended next step:
${analysisData.recommendedAction || "Review mitigation options and compare fairness tradeoffs."}
`;

    return `# FairLens AI Fairness Report

Generated at: ${generatedAt}
Dataset: ${datasetName}
Target column: ${target}
Sensitive features: ${sensitive}

## Executive Summary

${executiveSummary}

## Project Summary

${projectSummary}

## Audit Coverage

- Audit mode: ${auditScope}
- Sensitive attributes reviewed: ${sensitive}
- Outcome metric displayed by group: Predicted ${readableOutcomeLabel} rate
- Most affected group signal: ${mostAffectedGroupLabel}

## Baseline Findings

${beforeSummary}

Most affected group signal:
${analysisData.errorHeadline}

Interpretation:
${analysisData.errorText}

Group outcome rates before mitigation:
${formatGroupMetrics(analysisData.groupMetrics)}

## AI Insights

Generated by: ${insightsSource}
${insightsSource !== "gemini" && insightsError ? `Fallback reason: ${insightsError}` : ""}

${analysisData.insights.map((item) => `- ${item}`).join("\n")}

## Alerts

${analysisData.alerts.map((item) => `- ${item}`).join("\n")}

${mitigationSection}
`;
  }

  function downloadReport() {
    const reportContent = buildReportContent();
    const safeDatasetName = (analysisMeta.datasetName || "fairlens-report")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    const blob = new Blob([reportContent], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeDatasetName || "fairlens-report"}-fairness-report.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function getOutcomeRateLabel(targetColumnName, positiveLabel) {
    const cleanedTarget = String(targetColumnName || "")
      .replace(/_/g, " ")
      .trim();
    const cleanedPositive = String(positiveLabel || "").trim();

    if (cleanedPositive) {
      return `Predicted ${cleanedPositive} rate`;
    }
    if (cleanedTarget) {
      return `Predicted ${cleanedTarget} positive rate`;
    }
    return "Predicted positive outcome rate";
  }

  function getReadableOutcomeLabel(targetColumnName, positiveLabel) {
    const cleanedPositive = String(positiveLabel || "").trim();
    if (cleanedPositive) {
      return cleanedPositive;
    }

    const cleanedTarget = String(targetColumnName || "")
      .replace(/_/g, " ")
      .trim();
    return cleanedTarget || "positive outcome";
  }

  async function runAnalysis() {
    if (isAnalyzing || isApplyingFixes) {
      return;
    }

    setIsAnalyzing(true);
    setCurrentScreen("loading");
    setApiError("");
    setRequestMessage("Analyzing your dataset for fairness patterns...");
    setHasMitigated(false);

    if (!datasetFile) {
      window.setTimeout(() => {
        setAnalysisData(defaultBeforeData);
        setInsightsSource("demo");
        setInsightsError("");
        setRecommendationError("");
        setAnalysisMeta({
          datasetName: "Sample demo dataset",
          targetColumn: "Loan Approved",
          targetPositiveLabel: "Approved",
          detectedSensitiveFeatures: ["Gender", "Age", "Income"],
          targetNote: "",
          fairnessMetric: "Demographic Parity Difference",
        });
        setAfterData({
          score: 0.21,
          demographicParityDifference: 0.21,
          equalizedOddsDifference: 0.17,
          performance: {
            accuracy: 0.79,
            precision: 0.76,
            recall: 0.75,
          },
          groupMetrics: {
            Male: 69,
            Female: 63,
          },
          recommendedAction:
            "Validate the mitigated demo model on a holdout split and confirm the fairness lift persists.",
          recommendationSource: "demo",
        });
        setRequestMessage("");
        setIsAnalyzing(false);
        setCurrentScreen("dashboard");
      }, 1200);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("dataset", datasetFile);
      formData.append("target_column", targetColumn);
      formData.append("sensitive_features", JSON.stringify(sensitiveFeatures));

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const payload = await parseApiResponse(response);

      if (!response.ok) {
        throw new Error(payload.detail || payload.error || "Analysis failed.");
      }

      setAnalysisData({
        score: payload.bias_score,
        demographicParityDifference: toNumberOrNull(
          payload.demographic_parity_difference,
        ),
        equalizedOddsDifference: toNumberOrNull(
          payload.equalized_odds_difference,
        ),
        status: payload.severity,
        badge: payload.badge,
        performance: {
          accuracy: toNumberOrNull(payload?.performance?.accuracy),
          precision: toNumberOrNull(payload?.performance?.precision),
          recall: toNumberOrNull(payload?.performance?.recall),
        },
        groupMetrics: payload.group_metrics,
        errorHeadline: payload.error_headline,
        errorText: payload.error_text,
        insights: payload.insights,
        alerts: payload.alerts,
        recommendedAction: payload.recommended_action,
        recommendationSource:
          payload.recommendation_source ||
          payload.insights_source ||
          "rule-based",
      });
      setAnalysisMeta({
        datasetName: payload.dataset_name,
        targetColumn: payload.target_column,
        targetPositiveLabel: payload.target_positive_label || "",
        detectedSensitiveFeatures: payload.sensitive_features,
        targetNote: payload.target_note || "",
        fairnessMetric:
          payload.fairness_metric || "Demographic Parity Difference",
      });
      setInsightsSource(payload.insights_source || "rule-based");
      setInsightsError(payload.insights_error || "");
      setRecommendationError(payload.recommendation_error || "");
      setRequestMessage("");
      setCurrentScreen("dashboard");
    } catch (error) {
      setApiError(error.message || "Unable to analyze dataset right now.");
      setRequestMessage("");
      setCurrentScreen("upload");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function applyFixes() {
    if (isAnalyzing || isApplyingFixes) {
      return;
    }

    setIsApplyingFixes(true);
    setApiError("");
    setRequestMessage("Preparing dataset for mitigation...");
    setCurrentScreen("loading");

      try {
        if (!datasetFile) {
          setRequestMessage("Comparing mitigation strategies and recalculating impact...");
          const selectedCount = Object.values(fixes).filter(Boolean).length;
        setAfterData({
          score: selectedCount >= 2 ? 0.21 : selectedCount === 1 ? 0.36 : 0.58,
          demographicParityDifference:
            selectedCount >= 2 ? 0.21 : selectedCount === 1 ? 0.36 : 0.58,
          equalizedOddsDifference:
            selectedCount >= 2 ? 0.17 : selectedCount === 1 ? 0.29 : 0.43,
          performance:
            selectedCount >= 2
              ? { accuracy: 0.79, precision: 0.76, recall: 0.75 }
              : selectedCount === 1
                ? { accuracy: 0.8, precision: 0.76, recall: 0.73 }
                : { accuracy: 0.82, precision: 0.79, recall: 0.68 },
          groupMetrics:
            selectedCount >= 2
              ? { Male: 69, Female: 63 }
              : selectedCount === 1
                ? { Male: 70, Female: 56 }
                : { Male: 72, Female: 45 },
          recommendedAction:
            "Review the improved model on a fresh split and confirm fairness remains stable across the selected groups.",
          recommendationSource: "demo",
        });
        setRecommendationError("");
        setRequestMessage("");
        setIsApplyingFixes(false);
        setHasMitigated(true);
        setCurrentScreen("compare");
        return;
      }

      const formData = new FormData();
      formData.append("dataset", datasetFile);
      formData.append("target_column", targetColumn);
      formData.append("sensitive_features", JSON.stringify(sensitiveFeatures));
      formData.append("fixes", JSON.stringify(fixes));

      setRequestMessage(
        fixes.constraint
          ? "Training a fairness-aware model. This step can take a little longer..."
          : fixes.rebalance
            ? "Rebalancing groups and recalculating fairness outcomes..."
            : "Recomputing outcomes with the selected mitigation options...",
      );

      const response = await fetch("/api/mitigate", {
        method: "POST",
        body: formData,
      });

      const payload = await parseApiResponse(response);

      if (!response.ok) {
        throw new Error(
          payload.detail || payload.error || "Bias mitigation failed.",
        );
      }

      setAfterData({
        score: payload.bias_score,
        demographicParityDifference: toNumberOrNull(
          payload.demographic_parity_difference,
        ),
        equalizedOddsDifference: toNumberOrNull(
          payload.equalized_odds_difference,
        ),
        performance: {
          accuracy: toNumberOrNull(payload?.performance?.accuracy),
          precision: toNumberOrNull(payload?.performance?.precision),
          recall: toNumberOrNull(payload?.performance?.recall),
        },
        groupMetrics: payload.group_metrics,
        recommendedAction: payload.recommended_action,
        recommendationSource:
          payload.recommendation_source ||
          payload.insights_source ||
          "rule-based",
      });
      setRecommendationError(payload.recommendation_error || "");
      setRequestMessage("");
      setHasMitigated(true);
      setCurrentScreen("compare");
    } catch (error) {
      setApiError(error.message || "Unable to apply fixes right now.");
      setRequestMessage("");
      setCurrentScreen("fix");
    } finally {
      setIsApplyingFixes(false);
    }
  }

  function toggleFix(key) {
    setFixes((current) => ({
      ...current,
      [key]: !current[key],
    }));
  }

  function toggleSensitiveFeature(feature) {
    setSensitiveFeatures((current) =>
      current.includes(feature)
        ? current.filter((item) => item !== feature)
        : [...current, feature],
    );
  }

  const fairnessLift =
    analysisData.score > 0
      ? Math.round(
          ((analysisData.score - afterData.score) / analysisData.score) * 100,
        )
      : 0;
  const accuracyDelta =
    typeof afterData.performance?.accuracy === "number" &&
    typeof analysisData.performance?.accuracy === "number"
      ? (
          afterData.performance.accuracy - analysisData.performance.accuracy
        ).toFixed(3)
      : "N/A";
  const scoreBadgeClass =
    analysisData.badge === "Low"
      ? "safe"
      : analysisData.badge === "High"
        ? "danger"
        : "warning";
  const currentGroups = getGroupEntries(analysisData.groupMetrics);
  const beforeCompareGroups = getPrimaryGroups(analysisData.groupMetrics);
  const afterCompareGroups = getPrimaryGroups(afterData.groupMetrics);
  const comparisonGroups = getComparisonGroups(
    analysisData.groupMetrics,
    afterData.groupMetrics,
  );
  const selectedSensitiveFeatures =
    analysisMeta.detectedSensitiveFeatures || [];
  const isIntersectional = selectedSensitiveFeatures.length > 1;
  const sensitiveLabel = isIntersectional
    ? "intersectional groups"
    : selectedSensitiveFeatures[0] || "group";
  const primarySensitiveLabel =
    selectedSensitiveFeatures[0] || "selected feature";
  const outcomeRateLabel = getOutcomeRateLabel(
    analysisMeta.targetColumn,
    analysisMeta.targetPositiveLabel,
  );
  const recommendationSourceLabel =
    analysisData.recommendationSource === "gemini"
      ? "Recommended by Google Gemini"
      : analysisData.recommendationSource === "demo"
        ? "Demo recommendation"
        : "Recommended by the local fairness engine";
  const recommendedFeatureText =
    selectedSensitiveFeatures.length > 0
      ? selectedSensitiveFeatures.join(", ")
      : "Select one or more sensitive features during upload";
  const mostAffectedGroupEntry =
    currentGroups.length > 0
      ? [...currentGroups].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0]
      : null;
  const mostAffectedGroupLabel = mostAffectedGroupEntry?.[0] || "No clear group";
  const fairnessLiftLabel = `${Math.abs(fairnessLift)}%`;
  const accuracyDeltaValue =
    typeof afterData.performance?.accuracy === "number" &&
    typeof analysisData.performance?.accuracy === "number"
      ? afterData.performance.accuracy - analysisData.performance.accuracy
      : null;
  const accuracyDeltaLabel =
    typeof accuracyDeltaValue === "number"
      ? `${accuracyDeltaValue >= 0 ? "+" : ""}${accuracyDeltaValue.toFixed(3)}`
      : "N/A";
  const recommendedNextStep =
    hasMitigated && afterData.recommendedAction
      ? afterData.recommendedAction
      : analysisData.recommendedAction;

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">FL</div>
          <div>
            <p className="eyebrow">Responsible AI Workflow</p>
            <h1>FairLens AI</h1>
          </div>
        </div>

        <nav className="progress-nav" aria-label="Product flow">
          {navItems.map((item) => {
            const isActive =
              currentScreen === "loading"
                ? item.id === "upload"
                : item.id === currentScreen;

            return (
              <button
                key={item.id}
                className={`progress-pill${isActive ? " active" : ""}`}
                onClick={() => setCurrentScreen(item.id)}
                type="button"
              >
                {item.label}
              </button>
            );
          })}
        </nav>
      </header>

      <main>
        <section
          ref={refs.landing}
          className={`screen hero-screen${currentScreen === "landing" ? " screen-active" : ""}`}
        >
          <div className="hero-copy">
            <p className="eyebrow">Bias Detection Workspace</p>
            <h2>
              <span className="hero-title-strong">FairLens AI </span>
              <br></br>
              <span className="hero-title-soft">
                Detect Bias Before It Detects You
              </span>
            </h2>
            <p className="hero-text">
              Evaluate who your model serves well, where outcomes diverge across
              groups, and which mitigation path improves fairness with the least
              tradeoff.
            </p>
            <div className="hero-actions">
              <button
                className="primary-button"
                onClick={() => setCurrentScreen("upload")}
                type="button"
              >
                🚀 Start Analysis
              </button>
              <button
                className="secondary-button"
                onClick={runAnalysis}
                type="button"
              >
                📂 View Sample Demo
              </button>
            </div>
          </div>
        </section>

        <section
          ref={refs.upload}
          className={`screen${currentScreen === "upload" ? " screen-active" : ""}`}
        >
          <div className="section-heading">
            <p className="eyebrow">Step 2 - Upload Data</p>
            <h2>Upload data and choose fairness signals</h2>
            <p>
              Choose the decision outcome column and the demographic or group
              columns you want the fairness analysis to compare.
            </p>
          </div>

          {apiError ? <div className="error-banner">{apiError}</div> : null}

          <div className="upload-layout">
            <div className="panel upload-panel">
              <h3>Upload Dataset (CSV)</h3>
              <label className="dropzone" htmlFor="datasetFile">
                <input
                  id="datasetFile"
                  type="file"
                  accept=".csv"
                  onChange={async (event) => {
                    const file = event.target.files?.[0] || null;
                    setDatasetFile(file);
                    setSelectedFile(file?.name || "No file selected");
                    setApiError("");

                    if (!file) {
                      setAvailableColumns([]);
                      setAvailableSensitiveOptions([]);
                      setColumnSamples({});
                      setTargetColumn("");
                      setSensitiveFeatures([]);
                      return;
                    }

                    const { headers, samples } = await readCsvPreview(file);
                    if (headers.length > 0) {
                      setAvailableColumns(headers);
                      setColumnSamples(samples);
                      const suggestedTarget = suggestTargetColumn(
                        headers,
                        samples,
                      );
                      const resolvedTarget =
                        suggestedTarget || headers[0] || "";
                      setTargetColumn(resolvedTarget);

                      const detectedSensitive = suggestSensitiveFeatures(
                        headers,
                        resolvedTarget,
                        samples,
                      );
                      setAvailableSensitiveOptions(
                        buildSensitiveOptions(headers, resolvedTarget),
                      );
                      setSensitiveFeatures([]);
                      if (detectedSensitive.length > 0) {
                        setRequestMessage(
                          `Suggested sensitive features found: ${detectedSensitive.join(", ")}.`,
                        );
                      } else {
                        setRequestMessage("");
                      }

                      if (!suggestedTarget && resolvedTarget) {
                        setApiError(
                          "No binary outcome was auto-detected. You can still choose any target column manually and FairLens will adapt it for analysis.",
                        );
                      }
                    }
                  }}
                />
                <span className="drop-title">Drag and drop your dataset</span>
                <span className="drop-subtitle">or click to browse files</span>
                <span className="file-name">{selectedFile}</span>
              </label>
            </div>

            <div className="panel controls-panel">
              <div className="field-group">
                <label htmlFor="target-select">Outcome Column</label>
                <p className="field-help">
                  Choose the column that represents the final decision or result
                  you want to evaluate for fairness. This is usually a binary
                  outcome such as approved or rejected, hired or not hired, or{" "}
                  {"<="}50K and {">"}50K.
                </p>
                <select
                  id="target-select"
                  value={targetColumn}
                  onChange={(e) => {
                    const newTarget = e.target.value;
                    setTargetColumn(newTarget);
                    setAvailableSensitiveOptions(
                      buildSensitiveOptions(availableColumns, newTarget),
                    );
                    setSensitiveFeatures((current) =>
                      current.filter((feature) => feature !== newTarget),
                    );
                  }}
                  className="panel"
                >
                  <option value="">Select a target column</option>
                  {availableColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                      {isBinaryLike(columnSamples[col] || [])
                        ? " (binary recommended)"
                        : ""}
                    </option>
                  ))}
                </select>
                {isContinuousFeature(targetColumn, columnSamples, 6) ? (
                  <p className="field-help">
                    This column looks continuous (for example raw income). For
                    best fairness results, use a binary outcome column like
                    approved/rejected.
                  </p>
                ) : null}
              </div>

              <div className="field-group">
                <span>Sensitive Features (Demographic / Group Columns)</span>
                <p className="field-help">
                  Pick columns like gender, age, race, income band, or region so
                  FairLens can compare outcomes across groups.
                </p>
                {availableSensitiveOptions.length === 0 ? (
                  <p className="field-help">
                    Upload a CSV to load candidate sensitive columns from the
                    dataset.
                  </p>
                ) : null}
                {availableSensitiveOptions.map((feature) => (
                  <label key={feature} className="checkbox-row">
                    <input
                      checked={sensitiveFeatures.includes(feature)}
                      onChange={() => toggleSensitiveFeature(feature)}
                      type="checkbox"
                    />
                    <span>{feature}</span>
                  </label>
                ))}
              </div>

              <button
                className="primary-button"
                onClick={runAnalysis}
                type="button"
                disabled={
                  isAnalyzing ||
                  isApplyingFixes ||
                  (Boolean(datasetFile) &&
                    (!targetColumn || sensitiveFeatures.length === 0))
                }
              >
                {isAnalyzing ? "Analyzing..." : "🔍 Analyze Bias"}
              </button>
            </div>
          </div>
        </section>

        <section
          ref={refs.loading}
          className={`screen loading-screen${currentScreen === "loading" ? " screen-active" : ""}`}
        >
          <div className="loading-card">
            <div className="spinner" aria-hidden="true"></div>
            <p className="eyebrow">Step 3 - Analyze</p>
            <h2>Analyzing fairness patterns...</h2>
            <p>
              We are checking outcome rates, group error gaps, and protected
              attribute influence.
            </p>
            {requestMessage ? <p className="muted">{requestMessage}</p> : null}
          </div>
        </section>

        <section
          ref={refs.dashboard}
          className={`screen${currentScreen === "dashboard" ? " screen-active" : ""}`}
        >
          <div className="section-heading dashboard-heading">
            <div>
              <p className="eyebrow">Step 4 - View Report</p>
              <h2>Bias report dashboard</h2>
              <p>
                Dataset: {analysisMeta.datasetName || "Sample demo dataset"} |
                Target: {analysisMeta.targetColumn}
              </p>
              {analysisMeta.targetNote ? (
                <p>{analysisMeta.targetNote}</p>
              ) : null}
            </div>
            {hasMitigated ? (
              <div className="section-actions">
                <button
                  className="secondary-button"
                  onClick={downloadReport}
                  type="button"
                >
                  Export Report
                </button>
                <button
                  className="secondary-button"
                  onClick={() => setCurrentScreen("compare")}
                  type="button"
                >
                  Compare Results
                </button>
              </div>
            ) : (
              <div className="section-actions">
                <button
                  className="secondary-button"
                  onClick={downloadReport}
                  type="button"
                >
                  Export Report
                </button>
                <button
                  className="secondary-button"
                  onClick={() => setCurrentScreen("fix")}
                  type="button"
                >
                  Fix Bias
                </button>
              </div>
            )}
          </div>

          {hasMitigated ? (
            <div className="success-banner">
              Fixes applied successfully. Updated model bias score:{" "}
              <strong>{afterData.score.toFixed(2)}</strong>. Open comparison to
              see before vs after outcomes.
            </div>
          ) : null}

          <div className="dashboard-grid">
            <article className="panel score-panel">
              <p className="panel-label">Bias Score Card</p>
              <div className="score-wrap">
                <div>
                  <span className="score-value">
                    {analysisData.score.toFixed(2)}
                  </span>
                  <p className="score-status">{analysisData.status}</p>
                </div>
                <span className={`score-badge ${scoreBadgeClass}`}>
                  {analysisData.badge}
                </span>
              </div>
              <p className="muted">
                Primary fairness metric: {analysisMeta.fairnessMetric}. The
                severity also accounts for the observed group outcome gap.
              </p>
              <p className="muted">
                DP diff:{" "}
                {formatMetric(analysisData.demographicParityDifference)} | EO
                diff: {formatMetric(analysisData.equalizedOddsDifference)}
              </p>
            </article>

            <article className="panel summary-highlight-panel">
              <div className="panel-head">
                <p className="panel-label">Decision Summary</p>
                <span className="pill-soft">Stakeholder View</span>
              </div>
              <div className="summary-highlight-grid">
                <div className="summary-highlight-item">
                  <span>Risk Level</span>
                  <strong>{analysisData.status}</strong>
                </div>
                <div className="summary-highlight-item">
                  <span>Most Affected Group</span>
                  <strong>{mostAffectedGroupLabel}</strong>
                </div>
                <div className="summary-highlight-item">
                  <span>Fairness Lift</span>
                  <strong>{hasMitigated ? fairnessLiftLabel : "Pending mitigation"}</strong>
                </div>
                <div className="summary-highlight-item">
                  <span>Accuracy Delta</span>
                  <strong>{hasMitigated ? accuracyDeltaLabel : "Pending mitigation"}</strong>
                </div>
              </div>
              <p className="summary-narrative">
                {recommendedNextStep ||
                  "Review the fairness findings and choose a mitigation strategy before deployment."}
              </p>
            </article>

            <article className="panel chart-panel">
              <div className="panel-head">
                <p className="panel-label">Model Performance</p>
                <span className="pill-soft">Baseline</span>
              </div>
              <div className="bar-chart">
                <div className="bar-row">
                  <span>Accuracy</span>
                  <strong>
                    {formatMetric(analysisData.performance?.accuracy)}
                  </strong>
                </div>
                <div className="bar-row">
                  <span>Precision</span>
                  <strong>
                    {formatMetric(analysisData.performance?.precision)}
                  </strong>
                </div>
                <div className="bar-row">
                  <span>Recall</span>
                  <strong>
                    {formatMetric(analysisData.performance?.recall)}
                  </strong>
                </div>
              </div>
            </article>

            <article className="panel chart-panel">
              <div className="panel-head">
                <p className="panel-label">
                  {outcomeRateLabel} by{" "}
                  {isIntersectional ? "intersectional groups" : sensitiveLabel}
                </p>
                <span className="pill-soft">Parity Gap</span>
              </div>
              <div className="bar-chart">
                {currentGroups.map(([label, value], index) => (
                  <div key={label} className="bar-row">
                    <span>{label}</span>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${index % 2 === 0 ? "male-bar" : "female-bar"}`}
                        style={{ width: `${value}%` }}
                      ></div>
                    </div>
                    <strong>{value}%</strong>
                  </div>
                ))}
              </div>
              {isIntersectional ? (
                <p className="muted chart-note">
                  Multiple sensitive features were selected, so each bar shows a
                  combined group such as gender plus race or age band.
                </p>
              ) : (
                <p className="muted chart-note">
                  This chart compares predicted positive rates across{" "}
                  {primarySensitiveLabel}.
                </p>
              )}
            </article>

            <article className="panel chart-panel">
              <div className="panel-head">
                <p className="panel-label">Error Rate Comparison</p>
                <span className="pill-soft alert">Alert</span>
              </div>
              <div className="insight-block">
                <h3>{analysisData.errorHeadline}</h3>
                <p>{analysisData.errorText}</p>
              </div>
            </article>

            <article className="panel insights-panel">
              <p className="panel-label">AI Insights</p>
              <p className="muted">Generated by: {insightsSource}</p>
              {insightsSource !== "gemini" && insightsError ? (
                <p className="muted">Google AI fallback: {insightsError}</p>
              ) : null}
              <ul>
                {analysisData.insights.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>

            <article className="panel alerts-panel">
              <p className="panel-label">Alerts</p>
              <div className="alert-list">
                {analysisData.alerts.map((item) => (
                  <div key={item} className="alert-item">
                    {item}
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        <section
          ref={refs.fix}
          className={`screen${currentScreen === "fix" ? " screen-active" : ""}`}
        >
          <div className="section-heading">
            <p className="eyebrow">Step 5 - Fix Bias</p>
            <h2>Choose how to reduce bias</h2>
            <p>
              Apply fairness interventions, then re-run the analysis to measure
              the improvement.
            </p>
            {apiError ? <div className="error-banner">{apiError}</div> : null}
          </div>

          <div className="fix-layout">
            <div className="panel fix-options">
              <label className="toggle-card">
                <input
                  checked={fixes.sensitive}
                  onChange={() => toggleFix("sensitive")}
                  type="checkbox"
                />
                <span>
                  <strong>Remove sensitive feature</strong>
                  <small>
                    Exclude {primarySensitiveLabel} from model inputs where
                    possible
                  </small>
                </span>
              </label>

              <label className="toggle-card">
                <input
                  checked={fixes.rebalance}
                  onChange={() => toggleFix("rebalance")}
                  type="checkbox"
                />
                <span>
                  <strong>Rebalance dataset</strong>
                  <small>
                    Increase representation across lower-rate groups
                  </small>
                </span>
              </label>

              <label className="toggle-card">
                <input
                  checked={fixes.constraint}
                  onChange={() => toggleFix("constraint")}
                  type="checkbox"
                />
                  <span>
                    <strong>Apply fairness constraint</strong>
                    <small>
                      Optimize prediction quality while reducing parity gaps. This is the slowest option.
                    </small>
                  </span>
                </label>

              <button
                className="primary-button"
                onClick={applyFixes}
                type="button"
                disabled={isAnalyzing || isApplyingFixes}
              >
                {isApplyingFixes ? "Applying..." : "⚡ Apply Fix"}
              </button>
            </div>

            <aside className="panel recommendation-panel">
              <p className="panel-label">Recommended Action</p>
              <h3>
                {analysisData.recommendedAction ||
                  "Review the selected mitigation options"}
              </h3>
              <p>{recommendationSourceLabel}</p>
              <p className="muted">
                Sensitive features detected: {recommendedFeatureText}
              </p>
              {analysisData.recommendationSource === "gemini" ? (
                <p className="muted">
                  This recommendation is generated from the current fairness
                  metrics and group gaps using Google Gemini.
                </p>
              ) : recommendationError ? (
                <p className="muted">
                  Google AI fallback: {recommendationError}
                </p>
              ) : null}
            </aside>
          </div>
        </section>

        <section
          ref={refs.compare}
          className={`screen${currentScreen === "compare" ? " screen-active" : ""}`}
        >
          <div className="section-heading">
            <p className="eyebrow">Step 6 - Compare Results</p>
            <h2>Before vs after comparison</h2>
            <p>Review how the fairness profile changed after mitigation.</p>
            <div className="section-actions">
              <button
                className="secondary-button"
                onClick={downloadReport}
                type="button"
              >
                Export Report
              </button>
            </div>
            {isIntersectional ? (
              <p className="muted">
                Because you selected multiple sensitive features, the comparison
                highlights intersectional groups rather than a single
                demographic column.
              </p>
            ) : null}
          </div>

          <div className="compare-grid">
            <article className="panel compare-metrics-panel">
              <div className="panel-head">
                <p className="panel-label">Impact Summary</p>
                <span className="pill-soft">Before vs After</span>
              </div>
              <div className="compare-metrics-grid">
                <div className="metric-delta-card">
                  <span>Bias Score</span>
                  <strong>
                    {analysisData.score.toFixed(2)} → {afterData.score.toFixed(2)}
                  </strong>
                </div>
                <div className="metric-delta-card">
                  <span>DP Difference</span>
                  <strong>
                    {formatMetric(analysisData.demographicParityDifference)} →{" "}
                    {formatMetric(afterData.demographicParityDifference)}
                  </strong>
                </div>
                <div className="metric-delta-card">
                  <span>EO Difference</span>
                  <strong>
                    {formatMetric(analysisData.equalizedOddsDifference)} →{" "}
                    {formatMetric(afterData.equalizedOddsDifference)}
                  </strong>
                </div>
                <div className="metric-delta-card">
                  <span>Accuracy</span>
                  <strong>
                    {formatMetric(analysisData.performance?.accuracy)} →{" "}
                    {formatMetric(afterData.performance?.accuracy)}
                  </strong>
                </div>
              </div>
            </article>

            <article className="panel group-compare-panel">
              <div className="panel-head">
                <p className="panel-label">Group Fairness Comparison</p>
                <span className="pill-soft">Audit Coverage</span>
              </div>
              <p className="muted">
                Review how predicted positive outcome rates shifted across the audited groups after mitigation.
              </p>
              <div className="group-compare-list">
                {comparisonGroups.map((group) => (
                  <div key={group.label} className="group-compare-row">
                    <div className="group-compare-header">
                      <strong>{group.label}</strong>
                      <span>
                        {formatPercent(group.before)} → {formatPercent(group.after)}
                      </span>
                    </div>
                    <div className="group-compare-bars">
                      <div className="group-compare-track">
                        <span
                          className="group-compare-fill before-fill"
                          style={{ width: `${group.before}%` }}
                        ></span>
                      </div>
                      <div className="group-compare-track">
                        <span
                          className="group-compare-fill after-fill"
                          style={{ width: `${group.after}%` }}
                        ></span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel compare-panel">
              <span className="compare-label before">Before</span>
              <h3>Original Model</h3>
              <p className="compare-score">
                Bias: <strong>{analysisData.score.toFixed(2)}</strong>
              </p>
              <p className="muted">
                Accuracy: {formatMetric(analysisData.performance?.accuracy)} |
                DP: {formatMetric(analysisData.demographicParityDifference)} |
                EO: {formatMetric(analysisData.equalizedOddsDifference)}
              </p>
              <p className="compare-subtitle">
                Most affected{" "}
                {isIntersectional ? "groups" : primarySensitiveLabel} before
                mitigation
              </p>
              <div className="mini-chart">
                {beforeCompareGroups.map(([label, value]) => (
                  <div key={label} className="mini-bar">
                    <span>{label}</span>
                    <div>
                      <i style={{ width: `${value}%` }}></i>
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel compare-panel improved">
              <span className="compare-label after">After</span>
              <h3>Mitigated Model</h3>
              <p className="compare-score">
                Bias: <strong>{afterData.score.toFixed(2)}</strong>
              </p>
              <p className="muted">
                Accuracy: {formatMetric(afterData.performance?.accuracy)} | DP:{" "}
                {formatMetric(afterData.demographicParityDifference)} | EO:{" "}
                {formatMetric(afterData.equalizedOddsDifference)}
              </p>
              <p className="compare-subtitle">
                Highest positive-rate groups after mitigation
              </p>
              <div className="mini-chart">
                {afterCompareGroups.map(([label, value]) => (
                  <div key={label} className="mini-bar">
                    <span>{label}</span>
                    <div>
                      <i style={{ width: `${value}%` }}></i>
                    </div>
                  </div>
                ))}
              </div>
            </article>
          </div>

          <div className="panel summary-panel">
            Bias{" "}
            {afterData.score < analysisData.score ? "dropped" : "increased"}{" "}
            from {analysisData.score.toFixed(2)} to {afterData.score.toFixed(2)}
            , {fairnessLift > 0 ? "improving" : "changing"} the fairness score
            by {Math.abs(fairnessLift)}% with an accuracy delta of{" "}
            {accuracyDelta}.
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
