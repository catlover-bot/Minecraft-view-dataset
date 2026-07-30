# Completeness and repair burden

Completeness correlated with action count (rho=0.435), predicted voxel count (rho=0.422), predicted/GT voxel ratio (rho=0.428), delete cost (rho=0.426), replace cost (rho=0.424), and total repair cost (rho=0.411); add cost was negatively associated (rho=-0.428). OpenAI and Claude completeness scores were constant, while the within-model relationship was estimable for Gemini, so model composition and Gemini variation materially contribute.

After adjustment for description model, GT voxel count, bbox volume, material count, and description length with building-cluster-robust standard errors, the completeness coefficient remained 1.352 (95% CI [0.780, 1.924], p=3.619e-06). This is associational.

Normalization sensitivity was substantial: rho=0.411 under the original GT normalization, 0.345 under max(GT,pred), and -0.427 under GT+pred normalization. Thus A (association with greater generation/delete burden), C (model/Gemini contribution), D (normalization contribution), and E (adjusted association) are supported; B (fully explained by building complexity) is not supported.
