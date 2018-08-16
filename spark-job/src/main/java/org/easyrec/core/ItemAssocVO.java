/*
 * Copyright 2010 Research Studios Austria Forschungsgesellschaft mBH
 *
 * This file is part of easyrec.
 *
 * easyrec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * easyrec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with easyrec.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.easyrec.core;

import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * This class is a VO (valueobject/dataholder) for a SAT recommender database <code>ItemAssoc</code>.
 * All typed attributes use a different set of integer ids for each type.
 * <p/>
 * <p><b>Company:&nbsp;</b>
 * SAT, Research Studios Austria</p>
 * <p/>
 * <p><b>Copyright:&nbsp;</b>
 * (c) 2007</p>
 * <p/>
 * <p><b>last modified:</b><br/>
 * $Author: dmann $<br/>
 * $Date: 2011-12-20 15:22:22 +0100 (Di, 20 Dez 2011) $<br/>
 * $Revision: 18685 $</p>
 *
 * @author Roman Cerny
 */

public class ItemAssocVO<I extends Comparable<I>, T extends Comparable<T>>
        implements Cloneable, Serializable, Comparable<ItemAssocVO<I,T>> {
    /////////////////////////////////////////////////////////////////////////
    // constants
    private static final long serialVersionUID = -3231266117118582233L;

    // HINT: add a possibility to import item associations from a .CSV file with a given date (Mantis Issue: #582)
    /**
     * the number of columns a .CSV file must contain (tenantId, itemFromId, itemFromTypeId, assocTypeId, assocValue, itemToId, itemToTypeId, sourceTypeId, sourceInfo, viewTypeId, active)
     * Note: the attributes id and changeDate are left out, since they are automatically generated by the database
     */
    public static final int CSV_NUMBER_OF_COLUMNS = 11;

    ////////////////////////////////////////////////////////////////////////
    // members
    private Long id;

    private I tenant;

    private ItemVO<I,T> itemFrom;

    private T assocType;
    private Double assocValue;

    private ItemVO<I,T> itemTo;

    private T sourceType;
    private String sourceInfo;

    private T viewType;

    private Boolean active;

    private Date changeDate;

    // //////////////////////////////////////////////////////////////////////
    // methods
    public ItemAssocVO(I tenant, ItemVO<I,T> itemFrom, T assocType, Double assocValue, ItemVO<I,T> itemTo,
                       T sourceType, String sourceInfo, T viewType, Boolean active) {
        this(null, tenant, itemFrom, assocType, assocValue, itemTo, sourceType, sourceInfo, viewType, active, null);
    }

    public ItemAssocVO(Long id, I tenant, ItemVO<I,T> itemFrom, T assocType, Double assocValue,
                       ItemVO<I,T> itemTo, T sourceType, String sourceInfo, T viewType, Boolean active) {
        this(id, tenant, itemFrom, assocType, assocValue, itemTo, sourceType, sourceInfo, viewType, active, null);
    }

    public ItemAssocVO(I tenant, ItemVO<I,T> itemFrom, T assocType, Double assocValue, ItemVO<I,T> itemTo,
                       T sourceType, String sourceInfo, T viewType, Boolean active, Date changeDate) {
        this(null, tenant, itemFrom, assocType, assocValue, itemTo, sourceType, sourceInfo, viewType, active,
                changeDate);
    }

    public ItemAssocVO(Long id, I tenant, ItemVO<I,T> itemFrom, T assocType, Double assocValue,
                       ItemVO<I,T> itemTo, T sourceType, String sourceInfo, T viewType, Boolean active,
                       Date changeDate) {
        setId(id);
        setTenant(tenant);
        setItemFrom(itemFrom);
        setAssocType(assocType);
        setAssocValue(assocValue);
        setItemTo(itemTo);
        setSourceType(sourceType);
        setSourceInfo(sourceInfo);
        setViewType(viewType);
        setActive(active);
        setChangeDate(changeDate);
    }

    public Long getId() {
        return id;
    }

    public I getTenant() {
        return tenant;
    }

    public ItemVO<I,T> getItemFrom() {
        return itemFrom;
    }

    public T getAssocType() {
        return assocType;
    }

    public Double getAssocValue() {
        return assocValue;
    }

    public ItemVO<I,T> getItemTo() {
        return itemTo;
    }

    public T getSourceType() {
        return sourceType;
    }

    public String getSourceInfo() {
        return sourceInfo;
    }

    public T getViewType() {
        return viewType;
    }

    public Boolean isActive() {
        return active;
    }

    public Date getChangeDate() {
        return changeDate;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setTenant(I tenant) {
        this.tenant = tenant;
    }

    public void setItemFrom(ItemVO<I,T> itemFrom) {
        this.itemFrom = itemFrom;
    }

    public void setAssocType(T assocType) {
        this.assocType = assocType;
    }

    public void setAssocValue(Double assocValue) {
        this.assocValue = assocValue;
    }

    public void setItemTo(ItemVO<I,T> itemTo) {
        this.itemTo = itemTo;
    }

    public void setSourceType(T sourceType) {
        this.sourceType = sourceType;
    }

    public void setSourceInfo(String sourceInfo) {
        this.sourceInfo = sourceInfo;
    }

    public void setViewType(T viewType) {
        this.viewType = viewType;
    }

    public void setActive(Boolean active) {
        this.active = active;
    }

    public void setChangeDate(Date changeDate) {
        this.changeDate = changeDate;
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder(getClass().getSimpleName());
        s.append('@');
        s.append(Integer.toHexString(hashCode()));

        s.append("[id=");
        s.append(id);

        s.append(", tenant=");
        s.append(tenant);

        s.append(", itemFrom=");
        s.append(itemFrom);

        s.append(", assocType=");
        s.append(assocType);

        s.append(", assocValue=");
        s.append(assocValue);

        s.append(", itemTo=");
        s.append(itemTo);

        s.append(", sourceType=");
        s.append(sourceType);

        s.append(", sourceInfo='");
        s.append(sourceInfo);

        s.append("', viewType=");
        s.append(viewType);

        s.append(", active=");
        s.append(active);

        s.append(", changeDate=");
        s.append(changeDate);
        s.append("]");
        return s.toString();
    }

    public String toSqlValuesStr() {
        StringBuilder s = new StringBuilder("values(");

        s.append(tenant);
        s.append(", ");

        s.append(itemFrom.getItem());
        s.append(", ");

        s.append(itemFrom.getType());
        s.append(", ");

        s.append(assocType);
        s.append(", ");

        s.append(assocValue);
        s.append(", ");

        s.append(itemTo.getItem());
        s.append(", ");

        s.append(itemTo.getType());
        s.append(", ");

        s.append(sourceType);
        s.append(", ");

        s.append(sourceInfo);
        s.append(", ");

        s.append(viewType);
        s.append(", ");

        if (active != null) {
            s.append(active);
            s.append(", ");
        }

        if (changeDate == null){
            setChangeDate(new Date(System.currentTimeMillis()));
        }

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        s.append(sdf.format(changeDate));
        s.append(")");

        return s.toString();
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((active == null) ? 0 : active.hashCode());
        result = prime * result + ((assocType == null) ? 0 : assocType.hashCode());
        result = prime * result + ((assocValue == null) ? 0 : assocValue.hashCode());
        result = prime * result + ((changeDate == null) ? 0 : changeDate.hashCode());
        result = prime * result + ((id == null) ? 0 : id.hashCode());
        result = prime * result + ((itemFrom == null) ? 0 : itemFrom.hashCode());
        result = prime * result + ((itemTo == null) ? 0 : itemTo.hashCode());
        result = prime * result + ((sourceInfo == null) ? 0 : sourceInfo.hashCode());
        result = prime * result + ((sourceType == null) ? 0 : sourceType.hashCode());
        result = prime * result + ((tenant == null) ? 0 : tenant.hashCode());
        result = prime * result + ((viewType == null) ? 0 : viewType.hashCode());
        return result;
    }

    @SuppressWarnings("unchecked")
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        final ItemAssocVO<I,T> other = (ItemAssocVO<I,T>) obj;
        if (active == null) {
            if (other.active != null) return false;
        } else if (!active.equals(other.active)) return false;
        if (assocType == null) {
            if (other.assocType != null) return false;
        } else if (!assocType.equals(other.assocType)) return false;
        if (assocValue == null) {
            if (other.assocValue != null) return false;
        } else if (!assocValue.equals(other.assocValue)) return false;
        if (changeDate == null) {
            if (other.changeDate != null) return false;
        } else if (!changeDate.equals(other.changeDate)) return false;
        if (id == null) {
            if (other.id != null) return false;
        } else if (!id.equals(other.id)) return false;
        if (itemFrom == null) {
            if (other.itemFrom != null) return false;
        } else if (!itemFrom.equals(other.itemFrom)) return false;
        if (itemTo == null) {
            if (other.itemTo != null) return false;
        } else if (!itemTo.equals(other.itemTo)) return false;
        if (sourceInfo == null) {
            if (other.sourceInfo != null) return false;
        } else if (!sourceInfo.equals(other.sourceInfo)) return false;
        if (sourceType == null) {
            if (other.sourceType != null) return false;
        } else if (!sourceType.equals(other.sourceType)) return false;
        if (tenant == null) {
            if (other.tenant != null) return false;
        } else if (!tenant.equals(other.tenant)) return false;
        if (viewType == null) {
            if (other.viewType != null) return false;
        } else if (!viewType.equals(other.viewType)) return false;
        return true;
    }

    public int compareTo(ItemAssocVO<I,T> that) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;

        if (this == that) return EQUAL;

        if (this.assocValue < that.assocValue) return BEFORE;
        if (this.assocValue > that.assocValue) return AFTER;

        int comp = this.tenant.compareTo(that.tenant);
        if (comp != EQUAL) return comp;

        comp = this.itemFrom.compareTo(that.itemFrom);
        if (comp != EQUAL) return comp;

        comp = this.assocType.compareTo(that.assocType);
        if (comp != EQUAL) return comp;

        comp = this.itemTo.compareTo(that.itemTo);
        if (comp != EQUAL) return comp;

        comp = this.sourceType.compareTo(that.sourceType);
        if (comp != EQUAL) return comp;

        comp = this.viewType.compareTo(that.viewType);
        if (comp != EQUAL) return comp;

        if (!this.active && that.active) return AFTER;
        if (this.active && !that.active) return BEFORE;

        comp = this.changeDate.compareTo(that.changeDate);
        if (comp != EQUAL) return comp;

        //the following fields should be comparable even when null without NullPointerException
        if (this.id == null && that.id != null) return AFTER;
        if (this.id != null && that.id == null) return BEFORE;
        if (this.id != null && that.id != null) {
            if (this.id < that.id) return BEFORE;
            if (this.id > that.id) return AFTER;
        }

        if (this.sourceInfo == null && that.sourceInfo != null) return AFTER;
        if (this.sourceInfo != null && that.sourceInfo == null) return BEFORE;
        if (this.sourceInfo != null && that.sourceInfo != null) {
            comp = this.sourceInfo.compareTo(that.sourceInfo);
            if (comp != EQUAL) return comp;
        }

        assert this.equals(that) : "compareTo inconsistent with equals!";

        return EQUAL;
    }

    @SuppressWarnings("unchecked")
    @Override
    public ItemAssocVO<I,T> clone() throws CloneNotSupportedException {
        ItemAssocVO<I,T> clonedIAssocVO = (ItemAssocVO<I,T>) super.clone();
        clonedIAssocVO.setItemFrom(clonedIAssocVO.getItemFrom().clone());
        clonedIAssocVO.setItemTo(clonedIAssocVO.getItemTo().clone());
        return clonedIAssocVO;
    }
}