/*******************************************************************************
 * Copyright (c) 2011-2015 EclipseSource Muenchen GmbH and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 * Eugen Neufeld - initial API and implementation
 ******************************************************************************/
package org.eclipse.emfforms.internal.core.services.legacy;

import org.eclipse.emf.ecp.view.spi.context.GlobalViewModelService;
import org.eclipse.emf.ecp.view.spi.context.ViewModelContext;

public class TestGlobalViewModelService implements GlobalViewModelService {

	public TestGlobalViewModelService() {
		// intentionally left empty
	}

	@Override
	public void instantiate(ViewModelContext context) {
		context.putContextValue(getClass().getSimpleName(), getClass());
	}

	@Override
	public void dispose() {
		// intentionally left empty
	}

	@Override
	public int getPriority() {
		return 0;
	}

	@Override
	public void childViewModelContextAdded(ViewModelContext childContext) {
		// intentionally left empty
	}

}
