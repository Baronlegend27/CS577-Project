<hr style="border: none; border-top: 5px solid black;">
<h1 style = "text-align: center;">Neural Collaberative Filtering</h1>
<hr style="border: none; border-top: 1px solid black;">
<br>
<br>
<center>
    <strong style="font-size: 12pt;">Anonymous Author(s)</strong>
</center>
<p style="text-align: center; font-size: 10pt; line-height: 11pt;">
    Affiliation <br>
    Address <br>
    email
</p>
<br>
<center>
    <strong style="font-size: 12pt;">Abstract</strong>
</center>
<p style="text-align: justify; margin-left: 3pc; margin-right: 3pc; font-size: 10pt; line-height: 11pt;">
    Your abstract text goes here. This paragraph will be indented 1/2 inch on both sides, 
    with a vertical spacing of 11 points (leading). It is also in 10-point type, as specified. blalablabc labalbalablaba  lbalabalba lablabalbalabalablaba lbalbalabl abalbal abalbalablabalbalabl abalbalablabalab
</p>
<br>
<strong style="font-size: 12pt;">Background</strong>
<br>
<br>
<p style="text-align: justify; font-size: 10pt; line-height: 11pt;">
    Collaborative filtering is a method used in recommendation systems to suggest items to users based on the preferences and behaviors of other users. Basically assuming that if User $A$ liked something and User $B$ has similar taste to User $A$ then User $B$ will also like something User $A$ liked. 
</p>
<p style="text-align: justify; font-size: 10pt; line-height: 11pt;">
    Matrix factorization models are used to discover latent features that explain the patterns in user-item interactions. We basically have a matrix R where rows (m) represent the users and the columns (n) represent the items. We then want to factorize R into two smaller matrices (P) an $m * k$ where each row represents a user and the column represents a latent feature. For the item matrix $Q n * k$ where each row represents an item and each column represents a latent feature of the item. Then the factorization attempts to approximate the matrix R from $P * Q^T$. The latent factors or hidden features are the things which get learned that explain the observed interaction between users and items. 
</p>
<p style="text-align: justify; font-size: 10pt; line-height: 11pt;">
    Once the latent features are learned the dot product is used to predict (rank) the rating of between the users and items. WHere the interaction between a user and an item can be estimated by measuring how well the latent feature vectors for that user and that item match or align with each other (the dot product). When there is a high dot product it means that the latent features for a user and an item align well, while a low dot product means that the vectors do not align. 
</p>
<p style="text-align: justify; font-size: 10pt; line-height: 11pt;">
    However using the dot product does have some drawbacks. The dot product assumes that the user-item interactions are linearly dependent on the latent features. If two items are similar, the dot product assumes that the user’s preference for both items is simply the sum of their preferences for the individual latent features. The dot product assumes that user preferences can be captured by a single, linear combination of item attributes (represented by the item’s latent feature vector).
</p>
<br>
<strong style="font-size: 12pt;">Model Architecture</strong>
<br>
<br>
<p style="font-size: 10pt; line-height: 11pt;">
    Your abstract text goes here. This paragraph will be indented 1/2 inch on both sides, with a vertical spacing of 11 points (leading). It is also in 10-point type, as specified.
</p>
<br>
<strong style="font-size: 12pt;">Minimal CPU-Ready Working Example</strong>
<br>
<br>
<p style="font-size: 10pt; line-height: 11pt;">
    Your abstract text goes here. This paragraph will be indented 1/2 inch on both sides, 
    with a vertical spacing of 11 points (leading). It is also in 10-point type, as specified.
</p>
<br>
<strong style="font-size: 12pt;">Discussion</strong>
<br>
<br>
<p style="font-size: 10pt; line-height: 11pt;">
    Your abstract text goes here. This paragraph will be indented 1/2 inch on both sides, 
    with a vertical spacing of 11 points (leading). It is also in 10-point type, as specified.
</p>
<br>
<strong style="font-size: 12pt;">Contribution</strong>
<br>
<br>
<p style="font-size: 10pt; line-height: 11pt;">
    Your abstract text goes here. This paragraph will be indented 1/2 inch on both sides, 
    with a vertical spacing of 11 points (leading). It is also in 10-point type, as specified.
</p>
<br>
<strong style="font-size: 12pt;">References</strong>
<br>
<br>
<p style="font-size: 10pt; line-height: 11pt;">
    Any choice of citation style is acceptable as long as you are consistent It is permissible to reduce the font size to small (9 point) when listing the references.Note that the Reference section does not count towards the page limit.
</p>
<p style="font-size: 10pt; line-height: 11pt;">
    [1] Alexander, J.A. & Mozer, M.C. (1995) Template-based algorithms for connectionist rule extraction. In G.Tesauro, D.S. Touretzky and T.K. Leen (eds.), <i>Advances in Neural Information Processing Systems 7</i>,pp. 609–616. Cambridge, MA: MITPress.
</p>
<p style="font-size: 10pt; line-height: 11pt;">
    [2] Bower, J.M. & Beeman, D. (1995) <i>The Book of GENESIS: Exploring Realistic Neural Models with the GEneral NEural SImulation System.</i> New York: TELOS/Springer–Verlag.
</p>
<p style="font-size: 10pt; line-height: 11pt;">
    [3] Hasselmo, M.E., Schnell, E. & Barkai, E. (1995) Dynamics of learning and recall at excitatory recurrent synapses and cholinergic modulation in rat hippocampal region CA3. <i>Journal of Neuroscience</i> <strong>15</strong>(7):5249-5262.
</p>
<br>
<strong style="font-size: 12pt;">Appendix</strong>
<br>
<br>
<p style="font-size: 10pt; line-height: 11pt;">
    Optionally include supplemental material (complete proofs additional experiments and plots) in appendix. All such materials <strong>SHOULD be included in the main submission.<strong>
</p>